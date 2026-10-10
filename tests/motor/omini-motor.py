import os
import time
from motor.protocol import *
from motor.transport import create_transport


class LkMotor:
    """
    用于控制瓴控电机的串口控制类，封装了所有典型控制命令。
    支持：开环、闭环扭矩、速度、多圈位置、单圈位置、增量控制等。
    """
    def __init__(
        self,
        port: str,
        baudrate: int = 460800,
        motor_id: int = 1,
        timeout: float | None = None,
        backend: str | None = None,
    ):
        """
        初始化串口连接和电机 ID。
        """
        self.motor_id = motor_id
        default_timeout = 0.005 if os.name == "nt" else 0.02
        self.ser = create_transport(
            port=port,
            baudrate=baudrate,
            timeout=timeout if timeout is not None else default_timeout,
            backend=backend,
        )
        self.transport_name = getattr(self.ser, "backend_name", "unknown")
        if not self.ser.is_open:
            self.ser.open()
        self.position = None  # 多圈角度，单位弧度
        self.velocity = None  # 速度，单位 弧度/s
        self.torque = None    # 当前 Iq 电流，近似力矩
        self.KT = 0.09
        
        self.last_valid_position = None # 用于存储上一次有效的位置
        self.stuck_counter = 0          # 卡住计数器
        self.MAX_STUCK_COUNT = 5        # 允许数据连续卡住的最大次数
        self.refresh_ok = None
        self.refresh_inter_read_delay_s = 0.0005 if os.name == "nt" else 0.002
        
        self.refresh_attempts = 0
        self.refresh_successes = 0

    def close(self):
        if hasattr(self.ser, "close"):
            self.ser.close()

    def send_command(self, cmd: int, data: list[int] = [], expect_reply_len: int = 0) -> bytes:
        """
        构造、发送一条指令并读取应答，包含头部/数据段校验。
        """
        self.ser.reset_input_buffer()
        frame = build_frame(cmd, self.motor_id, data)
        self.ser.write(frame)
        # time.sleep(0.005)

        if expect_reply_len > 0:
            resp = self.ser.read(expect_reply_len)

            if len(resp) != expect_reply_len:
                raise MotorTimeoutError("Timeout or incomplete response")

            if resp[0] != 0x3E:
                raise InvalidHeaderError("Invalid frame header")

            if not verify_checksum(resp[5:]):
                raise ChecksumError("Invalid data checksum")

            return resp
        return b''
    def send_raw_command(self, cmd: int, data: list[int]):
        """
        发送不需要读取回应的快速控制命令（如 MIT 控制）
        """
        try:
            self.send_command(cmd=cmd, data=data, expect_reply_len=0)
        except Exception as e:
            print(f"[Motor ID {self.motor_id}] 快速命令失败: {e}")

    def enable(self):
        """命令 0x88：启动电机"""
        self.send_command(0x88)

    def disable(self):
        """命令 0x80：关闭电机"""
        self.send_command(0x80)

    def stop(self):
        """命令 0x81：立即停止电机（停止控制输出）"""
        self.send_command(0x81)

    def clear_error(self):
        """命令 0x9B：清除错误位"""
        self.send_command(0x9B)

    def set_zero_ram(self):
        """命令 0x19：设置当前位置为零点（断电失效）"""
        self.send_command(0x19)

    def set_zero_rom(self):
        """命令 0x19：持久化零点（部分版本支持 ROM）"""
        self.send_command(0x19)

    def clear_turn_count(self):
        """命令 0x93：清除圈数信息（恢复为单圈）"""
        self.send_command(0x93)

    def read_status_1(self):
        """命令 0x9A：读取状态1（温度、电压、运行状态等）"""
        resp = self.send_command(0x9A, [], expect_reply_len=13)
        return parse_status1(resp[5:])

    def read_status_2(self):
        """命令 0x9C：读取状态2（Iq、电流、速度、编码器）"""
        resp = self.send_command(0x9C, [], expect_reply_len=13)
        return parse_status2(resp[5:])

    def read_encoder(self):
        """命令 0x90：读取编码器值、原始编码值与偏移"""
        resp = self.send_command(0x90, [], expect_reply_len=12)
        return parse_encoder(resp[5:])

    def read_multi_turn_angle(self):
        resp = self.send_command(0x92, [], expect_reply_len=14)
        motor_angle_deg = parse_angle64(resp[5:13]) / 10.0
        radian = degree_to_radian(motor_angle_deg)
        # print(f"[Motor ID {self.motor_id}] 多圈角度解析: {motor_angle_deg:+.2f}°")
        return radian

    def read_single_turn_angle(self):
        """命令 0x94：读取单圈角度（单位：0.01°，4字节）"""
        resp = self.send_command(0x94, [], expect_reply_len=10)
        # 单圈是 4 字节，单位 0.01°
        circle_angle_raw = int.from_bytes(resp[5:9], byteorder='little', signed=False)
        circle_angle_deg = circle_angle_raw / 100.0
        return circle_angle_deg

    def set_open_loop(self, power: int):
        """
        命令 0xA0：开环控制，输入功率值（-850~850）
        """
        bytes_ = list(power.to_bytes(2, 'little', signed=True))
        self.send_command(0xA0, bytes_)

    def set_torque(self, iq: float):
        """
        命令 0xA1：扭矩环控制，输入目标电流Iq（单位：A）
        """
        try:
            iq_scaled = iq * (2048 / 33.0)
            iq_int = int(round(iq_scaled))
            iq_int = max(-2048, min(2047, iq_int))
            bytes_ = list(iq_int.to_bytes(2, 'little', signed=True))
            self.send_command(0xA1, bytes_)
        except Exception as e:
            print("发送扭矩失败")

    def set_torque_nm(self, torque: float, kt: float = 0.09):
        iq = torque / kt
        self.set_torque(iq)


    def set_speed(self, speed_dps: float):
        """
        命令 0xA2：速度环控制，单位 deg/s，内部以 0.01°/s 表示
        """
        val = int(speed_dps * 1000)
        bytes_ = list(val.to_bytes(4, 'little', signed=True))
        self.send_command(0xA2, bytes_)

    def move_to_position(self, angle_deg: float):
        """
        命令 0xA3：位置环控制（多圈），单位为 0.01°
        """
        val = int(angle_deg * 100)
        bytes_ = list(val.to_bytes(8, 'little', signed=True))
        self.send_command(0xA3, bytes_)

    def move_to_position_with_speed(self, angle_deg: float, speed_dps: float):
        """
        命令 0xA4：位置+速度环控制（多圈）
        """
        a = int(angle_deg * 100).to_bytes(8, 'little', signed=True)
        s = int(speed_dps * 100).to_bytes(4, 'little')
        self.send_command(0xA4, list(a + s))

    def move_single_circle(self, angle_deg: float, clockwise: bool):
        """
        命令 0xA5：单圈位置控制（不带速度）
        - clockwise: True=顺时针, False=逆时针
        """
        direction = 0x00 if clockwise else 0x01
        val = int(angle_deg * 100)
        a = list(val.to_bytes(2, 'little'))
        self.send_command(0xA5, [direction] + a + [0x00])

    def move_single_circle_with_speed(self, angle_deg: float, clockwise: bool, speed_dps: float):
        """
        命令 0xA6：单圈位置+速度控制
        """
        direction = 0x00 if clockwise else 0x01
        a = int(angle_deg * 100).to_bytes(2, 'little')
        s = int(speed_dps * 100).to_bytes(4, 'little')
        self.send_command(0xA6, [direction] + list(a) + [0x00] + list(s))

    def move_incremental(self, angle_delta_deg: float):
        """
        命令 0xA7：增量移动（相对位移，单位 0.01°）
        """
        val = int(angle_delta_deg * 100)
        self.send_command(0xA7, list(val.to_bytes(4, 'little', signed=True)))

    def move_incremental_with_speed(self, angle_delta_deg: float, speed_dps: float):
        """
        命令 0xA8：增量移动 + 速度控制
        """
        a = int(angle_delta_deg * 100).to_bytes(4, 'little', signed=True)
        s = int(speed_dps * 100).to_bytes(4, 'little')
        self.send_command(0xA8, list(a + s))

    def read_param(self, param_id: int):
        """
        命令 0x40：读取设定参数
        - 返回值为 6 字节参数体（根据 ID 解释）
        """
        data = [param_id, 0x00]
        resp = self.send_command(0x40, data, expect_reply_len=13)
        return resp[6:-1]

    def write_param_ram(self, param_id: int, param_data: list[int]):
        """
        命令 0x42：将参数写入 RAM（掉电失效）
        """
        assert len(param_data) == 6
        data = [param_id] + param_data
        self.send_command(0x42, data)

    def write_param_rom(self, param_id: int, param_data: list[int]):
        """
        命令 0x44：将参数写入 ROM（掉电保留）
        """
        assert len(param_data) == 6
        data = [param_id] + param_data
        self.send_command(0x44, data)

    def getPosition(self):
        return self.position

    def getVelocity(self):
        return self.velocity

    def getTorque(self):
        return self.torque

    # def refresh(self):
    #     """
    #     刷新当前电机状态，更新 self.position / velocity / torque。
    #     使用单圈角度（单位：°）
    #     """
    #     try:
    #         self.position = self.read_multi_turn_angle()
    #         # time.sleep(0.01)
    #     except Exception as e:
    #         print(f"[Motor ID {self.motor_id}] 读取位置失败: {e} --------------------------------")

    #     time.sleep(0.002)

    #     try:
    #         status = self.read_status_2()
    #         # print("***********************************************************************")
    #         # print(status)
    #         vel_deg_per_sec = status.get("speed_dps", 0.0) / 10.0
    #         self.velocity = degree_to_radian(vel_deg_per_sec)
    #         iq_control = status.get("iq_or_power", 0.0)
    #         # print(f"IQ_Control={iq_control:+.2f} ")
    #         iq = iq_control / 2048.00 * 33.0
    #         # print(f"iq={iq:+.2f}")

    #         self.torque = iq * self.KT * 10
    #     except Exception as e:
    #         print("[Motor ID {self.motor_id}] 读取速度失败: {e} -------------------------------")
    
    def refresh(self):
        """
        刷新当前电机状态，并检查数据是否卡住。
        """
        
        self.refresh_attempts += 1
        
        # 先将当前状态置为无效
        self.position = None
        self.velocity = None
        self.torque = None
        # self.refresh_ok = False

        try:
            current_pos = self.read_multi_turn_angle()
            if self.refresh_inter_read_delay_s > 0:
                time.sleep(self.refresh_inter_read_delay_s)
            status = self.read_status_2()

            if self.last_valid_position is not None and abs(current_pos - self.last_valid_position) < 1e-6:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            
            self.position = current_pos
            self.last_valid_position = current_pos

            vel_deg_per_sec = status.get("speed_dps", 0.0) / 10.0
            self.velocity = degree_to_radian(vel_deg_per_sec)
            
            iq_control = status.get("iq_or_power", 0.0)
            iq = iq_control / 2048.00 * 33.0
            self.torque = iq * self.KT * 10
            # return True
            self.refresh_ok = True
            self.refresh_successes += 1

        except Exception as e:
            print(f"[Motor ID {self.motor_id}] 刷新状态失败: {e}")
            self.refresh_ok = False
            
    def get_failure_percentage(self) -> float:
        if self.refresh_attempts == 0:
            return 0.0
        
        failures = self.refresh_attempts - self.refresh_successes
        failure_rate = (failures / self.refresh_attempts) * 100.0
        
        return failure_rate

    def read_device_info(self) -> dict:
        """
        读取电机型号/驱动版本等设备信息（使用 0x12 命令）
        """
        # 构造请求帧
        header = [0x3E, 0x12, self.motor_id, 0x00]
        checksum = sum(header) & 0xFF
        frame = bytes(header + [checksum])
        self.ser.write(frame)
        time.sleep(0.05)

        # 期望返回：5字节帧头 + 58字节数据 + 1字节数据校验 = 64字节
        resp = self.ser.read(64)
        if len(resp) != 64:
            raise IOError(f"响应长度错误，仅收到 {len(resp)} 字节")

        if resp[0] != 0x3E or resp[1] != 0x12 or resp[2] != self.motor_id or resp[3] != 0x3A:
            raise ValueError("帧头验证失败，非合法设备信息帧")

        # 校验帧头
        expected_header_checksum = sum(resp[0:4]) & 0xFF
        if resp[4] != expected_header_checksum:
            raise ValueError("帧头校验失败")

        data = resp[5:63]  # 正确：取 58 字节
        data_checksum = sum(data) & 0xFF
        if resp[63] != data_checksum:
            raise ValueError("数据校验失败")

        def extract_string(segment: bytes) -> str:
            return segment.split(b'\x00')[0].decode('ascii', errors='ignore').strip()

        return {
            "driver_name": extract_string(data[0:20]),
            "motor_name": extract_string(data[20:40]),
            "motor_id": extract_string(data[40:52]),
            "hardware_version": int.from_bytes(data[52:54], "little"),
            "motor_version": int.from_bytes(data[54:56], "little"),
            "firmware_version": int.from_bytes(data[56:58], "little"),
        }

    def apply_mit_control(self, q_desired, dq_desired, kp, kd, torque_offset=0.0):
        """
        MIT控制，使用期望位置、速度和力矩进行控制。
        参数单位：
            q_desired      - 期望位置（°）
            dq_desired     - 期望速度（°/s）
            kp             - 位置环增益
            kd             - 速度环增益
            torque_offset  - 期望输出力矩（Nm）
        """

        Q_MAX = 360.0
        DQ_MAX = 2000.0 * 100
        TAU_MAX = 33.0
        IQ_MAX = 2048

        q_uint = float_to_uint(q_desired, -Q_MAX, Q_MAX, 16)
        dq_scaled = dq_desired * 100 
        dq_uint = float_to_uint(dq_scaled, -DQ_MAX, DQ_MAX, 12)
        kp_uint = float_to_uint(kp, 0, 500, 12)
        kd_uint = float_to_uint(kd, 0, 5, 12)

        KT = 1.1
        iq = torque_offset / KT
        iq = max(-33, min(33, iq))
        tau_uint = float_to_uint(iq, -33.0, 33.0, 12)

        buf = [0]*8
        buf[0] = (q_uint >> 8) & 0xFF
        buf[1] = q_uint & 0xFF
        buf[2] = (dq_uint >> 4) & 0xFF
        buf[3] = ((dq_uint & 0xF) << 4) | ((kp_uint >> 8) & 0xF)
        buf[4] = kp_uint & 0xFF
        buf[5] = (kd_uint >> 4) & 0xFF
        buf[6] = ((kd_uint & 0xF) << 4) | ((tau_uint >> 8) & 0xF)
        buf[7] = tau_uint & 0xFF

        self.send_raw_command(cmd=0xA8, data=buf)

    def is_valid(self):
        is_data_read_ok = (
            self.position is not None and
            self.velocity is not None and
            self.torque is not None
        )
        
        is_not_stuck = self.stuck_counter < self.MAX_STUCK_COUNT

        # if not is_not_stuck:
        #     print(f"[警告][Motor ID {self.motor_id}] 检测到数据卡住，当前读数被视为无效。")

        return is_data_read_ok and is_not_stuck

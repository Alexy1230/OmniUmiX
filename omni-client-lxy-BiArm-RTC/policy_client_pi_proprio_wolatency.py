"""Sequential no-latency-compensation client for OpenPI proprio checkpoints."""

import policy_client_pi_wolatency as implementation


if __name__ == '__main__':
    implementation.PROPRIOCEPTION = True
    implementation.main()

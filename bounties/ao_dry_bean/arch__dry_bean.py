import ao_arch as ar

FEATURE_COUNT = 16
BITS_PER_FEATURE = 8
OUTPUT_BITS = 3

description = "UCI Dry Bean multiclass benchmark"
arch_i = [BITS_PER_FEATURE for _ in range(FEATURE_COUNT)]
arch_z = [OUTPUT_BITS]
arch_c = []
connector_function = "forward_full_conn"

Arch = ar.Arch(
    arch_i,
    arch_z,
    arch_c,
    connector_function,
    description,
)

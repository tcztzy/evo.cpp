#include <string.h>

#include <evo/evo.h>

int main(void) {
  evo_batch *batch = 0;
  const int valid = evo_abi_version() == EVO_ABI_VERSION_CURRENT &&
                    strcmp(evo_status_name(EVO_STATUS_OK), "ok") == 0 &&
                    evo_batch_create(1, &batch) == EVO_STATUS_OK && batch != 0;
  evo_batch_free(batch);
  return valid ? 0 : 1;
}

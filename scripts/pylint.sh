#!/bin/bash

# megatron/core/models/retro/data/
# megatron/core/models/retro/model/

pylint \
  --disable=duplicate-code \
  --disable=abstract-method \
  --disable=too-many-instance-attributes \
  --disable=redefined-builtin \
  --disable=unused-argument \
  --disable=unused-variable \
  --disable=unused-import \
  --disable=line-too-long \
  --disable=arguments-renamed \
  --disable=no-member \
  --disable=too-many-arguments \
  --disable=cyclic-import \
  --disable=wrong-import-order \
  --disable=consider-using-f-string \
  --disable=broad-exception-raised \
  --disable=broad-exception-caught \
  --disable=unreachable \
  --disable=no-name-in-module \
  --disable=import-error \
  --disable=arguments-differ \
  --disable=no-value-for-parameter \
  --disable=import-outside-toplevel \
  --disable=too-many-locals \
  --disable=consider-using-sys-exit \
  --disable=too-many-statements \
  --disable=too-many-branches \
  --disable=chained-comparison \
  --disable=consider-using-generator \
  --disable=bare-except \
  --disable=attribute-defined-outside-init \
  --disable=unspecified-encoding \
  --disable=raise-missing-from \
  --disable=undefined-variable \
  --disable=relative-beyond-top-level \
  --disable=no-else-return \
  megatron/core/models/retro/model/
#   --missing-docstring \
#   --no-docstring-rgx=__.*__ \

# eof
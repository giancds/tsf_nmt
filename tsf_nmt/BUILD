# Description:
# Example neural translation models.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# __init__.py
py_library(
    name = "package",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":attention",
        ":build_ops",
        ":cells",
        ":content_functions",
        ":data_utils",
        ":decoders",
        ":encoders",
        ":lm_models",
        ":nmt_models",
        ":train_ops",
        ":translate_ops"
    ],
)

# attention.py
py_library(
    name = "attention",
    srcs = [
        "attention.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cells",
        ":content_functions",
    ],
)

# build_ops.py
py_library(
    name = "build_ops",
    srcs = [
        "build_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":attention",
        ":content_functions",
        ":decoders",
        ":lm_models",
        ":nmt_models",
    ],
)

# cells.py
py_library(
    name = "cells",
    srcs = [
        "cells.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
    ],
)

# content_functions.py
py_library(
    name = "content_functions",
    srcs = [
        "content_functions.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cells",
    ],
)

# data_utils.py
py_library(
    name = "data_utils",
    srcs = [
        "data_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [],
)

# decoders.py
py_library(
    name = "decoders",
    srcs = [
        "decoders.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":attention",
        ":cells",
        ":content_functions",
    ],
)

# encoders.py
py_library(
    name = "encoders",
    srcs = [
        "encoders.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
    ],
)

# lm_models.py
py_library(
    name = "lm_models",
    srcs = [
        "lm_models.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cells",
        ":data_utils",
    ],
)

# nmt_models.py
py_library(
    name = "nmt_models",
    srcs = [
        "nmt_models.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cells",
        ":data_utils",
        ":encoders",
        ":optimization_ops",
    ],
)

# optimization_ops.py
py_library(
    name = "optimization_ops",
    srcs = [
        "optimization_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
    ],
)

# train_lm.py
py_library(
    name = "train_lm",
    srcs = [
        "train_lm.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":train_ops",
    ],
)

# train_ops.py
py_library(
    name = "train_ops",
    srcs = [
        "train_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":build_ops",
        ":data_utils",
    ],
)

# translate_ops.py
py_library(
    name = "translate_ops",
    srcs = [
        "translate_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":build_ops",
        ":data_utils",
    ],
)

### BINARY ###
py_binary(
    name = "translate_global_attention",
    srcs = [
        "translate_global_attention.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":attention",
        ":content_functions",
        ":train_ops",
        ":translate_ops",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tsf_nmt:__subpackages__"],
)
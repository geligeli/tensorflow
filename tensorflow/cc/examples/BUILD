load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_copts")
# "tf_copts",
# "tf_cuda_library",
# "tf_custom_op_library",
# "tf_kernel_library",
# )
#tf_cc_binary(
#    name = "example",
#    srcs = ["example.cc"],
#    deps = [
#        "//tensorflow/cc:cc_ops",
#        "//tensorflow/cc:client_session",
#        "//tensorflow/core:tensorflow",
#    ],
#)

cc_proto_library(
    name = "snake_proto_cc",
    srcs = ["snake.proto"],
    use_grpc_plugin = True,
)

cc_library(
    name = "snake",
    srcs = ["snake.cc"],
    hdrs = ["snake.h"],
    copts = tf_copts() + [
        "-Werror",
        "-Wextra",
        "-Wall",
    ],
    deps = [
        ":snake_proto_cc",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_library(
    name = "mcts",
    srcs = ["mcts.cc"],
    hdrs = ["mcts.h"],
    copts = tf_copts() + [
        "-Werror",
        "-Wextra",
        "-Wall",
    ],
    deps = [
        ":snake",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_test(
    name = "snake_test",
    timeout = "short",
    testonly = 1,
    srcs = ["snake_test.cc"],
    deps = [
        ":snake",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "mcts_test",
    timeout = "short",
    testonly = 1,
    srcs = ["mcts_test.cc"],
    deps = [
        ":mcts",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:logging",
    ],
)

cc_binary(
    name = "snake_main",
    srcs = ["snake_main.cc"],
    deps = [
        ":snake",
        ":snake_proto_cc",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
    ],
)

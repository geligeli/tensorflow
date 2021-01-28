load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("//tensorflow:tensorflow.bzl", "lrt_if_needed", "tf_cc_binary", "tf_cc_test", "tf_copts")
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
    deps = [
        "//tensorflow/core:protos_all_cc",
    ],
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
        ":mcts_node",
        ":snake",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_library(
    name = "mcts_prediction_batching",
    hdrs = ["mcts_prediction_batching.h"],
    copts = tf_copts() + [
        "-Werror",
        "-Wextra",
        "-Wall",
        "-fexceptions",
    ],
    deps = [
        "@boost//:coroutine2",
        "@boost//:fiber",
    ],
)

cc_library(
    name = "network",
    hdrs = ["network.h"],
    copts = tf_copts() + [
        "-Werror",
        "-Wextra",
        "-Wall",
    ],
    deps = [
        ":snake",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:tag_constants",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_library(
    name = "mcts_node",
    hdrs = ["mcts_node.h"],
    copts = tf_copts() + [
        "-Werror",
        "-Wextra",
        "-Wall",
    ],
    deps = [
        ":snake",
        "//tensorflow/core/platform:logging",
    ],
)

cc_library(
    name = "az_mcts",
    hdrs = ["az_mcts.h"],
    copts = tf_copts() + [
        "-Werror",
        "-Wextra",
        "-Wall",
    ],
    deps = [
        ":mcts_node",
        ":network_fiber_batch",
        ":snake",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:tag_constants",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_library(
    name = "network_fiber_batch",
    hdrs = ["network_fiber_batch.h"],
    copts = tf_copts() + [
        "-Werror",
        "-Wextra",
        "-Wall",
        "-fexceptions",
    ],
    deps = [
        ":mcts_prediction_batching",
        ":network",
    ],
)

cc_library(
    name = "self_play",
    hdrs = ["self_play.h"],
    copts = tf_copts() + [
        "-Werror",
        "-Wextra",
        "-Wall",
        "-fexceptions",
    ],
    deps = [
        ":az_mcts",
        ":network",
        ":network_fiber_batch",
        ":snake",
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
    srcs = ["mcts_test.cc"],
    extra_copts = [
        "-fexceptions",
    ],
    # linkstatic = 1,
    deps = [
        ":mcts",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:tag_constants",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:logging",
        "@boost//:coroutine2",
        "@boost//:fiber",
    ],
)

tf_cc_test(
    name = "az_mcts_test",
    srcs = ["az_mcts_test.cc"],
    extra_copts = [
        "-fexceptions",
    ],
    linkstatic = 1,
    deps = [
        ":az_mcts",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:tag_constants",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:logging",
        "@boost//:coroutine2",
        "@boost//:fiber",
    ],
)

tf_cc_test(
    name = "self_play_test",
    srcs = ["self_play_test.cc"],
    extra_copts = [
        "-fexceptions",
    ],
    linkstatic = 1,
    deps = [
        ":self_play",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
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

cc_library(
    name = "replay_buffer",
    srcs = ["replay_buffer.cc"],
    hdrs = ["replay_buffer.h"],
    deps = [
        ":snake_proto_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

tf_cc_binary(
    name = "replay_buffer_main",
    srcs = ["replay_buffer_main.cc"],
    deps = [
        ":replay_buffer",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_binary(
    name = "self_play_main",
    srcs = ["self_play_main.cc"],
    copts = tf_copts() + [
        "-fexceptions",
    ],
    linkopts = ["-l:libtcmalloc.so.4"] + lrt_if_needed(),
    deps = [
        ":self_play",
        ":snake_proto_cc",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
    ],
)

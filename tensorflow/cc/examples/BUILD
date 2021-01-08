load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")
load("@rules_cc//cc:defs.bzl", "cc_library")

tf_cc_binary(
    name = "example",
    srcs = ["example.cc"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "snake",
    hdrs = ["snake.h"],
    srcs = ["snake.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ]
)

tf_cc_test(
    name = "snake_test",
    srcs = ["snake_test.cc"],
    testonly = 1,
    deps = [
        ":snake",
        "@com_google_googletest//:gtest_main",
    ]
)


tf_cc_binary(
    name = "snake_main",
    srcs = ["snake_main.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
    ],
)

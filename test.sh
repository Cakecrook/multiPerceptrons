#!/bin/bash

function test() {
    python source.py mnist_train_0_1.csv mnist_test_0_1.csv > output.txt
    cat output.txt
}

function updateTest() {
    source test.sh
}
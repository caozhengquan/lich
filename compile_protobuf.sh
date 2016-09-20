#!/bin/bash

set -x

protoc --cpp_out=. ./lich/proto/*.proto

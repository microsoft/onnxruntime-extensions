#!/bin/bash

# Azure ops dependencies
if [[ "$1" == "many64" ]]; then
  yum -y install openssl-devel
elif [[ "$1" == "many86" ]]; then
  yum -y install openssl-devel
else # for musllinux
  apk add openssl-dev
fi

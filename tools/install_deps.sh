#!/bin/bash

if [[ "$OCOS_ENABLE_AZURE" == "1" ]]
then
   if [[ "$1" == "many64" ]]; then
     yum -y install openssl-devel
   elif [[ "$1" == "many86" ]]; then
     yum -y install openssl-devel
   else # for musllinux
     apk add openssl-dev
   fi
fi

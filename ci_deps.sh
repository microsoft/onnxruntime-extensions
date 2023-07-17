if [ "$OCOS_ENABLE_AZURE" == "1" ]
then
   type="$1"
   if [[ "$type" == "many64" ]]; then
     yum -y install openssl openssl-devel wget && wget https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz && tar zxvf v1.1.0.tar.gz && cd rapidjson-1.1.0 && mkdir build && cd build && cmake .. && cmake --install . && cd ../.. && git clone https://github.com/triton-inference-server/client.git --branch r23.05 ~/client && ln -s ~/client/src/c++/library/libhttpclient.ldscript /usr/lib64/libhttpclient.ldscript
   elif [[ "$type" == "many86" ]]; then
     yum -y install openssl openssl-devel wget && wget https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz && tar zxvf v1.1.0.tar.gz && cd rapidjson-1.1.0 && mkdir build && cd build && cmake .. && cmake --install . && cd ../.. && git clone https://github.com/triton-inference-server/client.git --branch r23.05 ~/client && ln -s ~/client/src/c++/library/libhttpclient.ldscript /opt/rh/devtoolset-10/root/usr/lib/libhttpclient.ldscript
   else # for musllinux
     apk add openssl-dev wget && wget https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz && tar zxvf v1.1.0.tar.gz && cd rapidjson-1.1.0 && mkdir build && cd build && cmake .. && cmake --install . && cd ../.. && git clone https://github.com/triton-inference-server/client.git --branch r23.05 ~/client && ln -s ~/client/src/c++/library/libhttpclient.ldscript /usr/lib/libhttpclient.ldscript
   fi
fi
~

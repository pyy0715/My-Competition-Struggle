#!/bin/zsh

cd ./Dataset

echo $PWD

echo 'TrainData_02 Download'
wget 'http://arena.kakaocdn.net/kakao_arena/shopping/train.chunk.02?TWGServiceId=kakao_arena&Expires=1615302068&Signature=qf0KSifAx5uUIRTjMm/ajGwXf34%3D&AllowedIp=110.9.51.148&download' -O train.chunk.02

echo 'TrainData_03 Download'
wget 'http://arena.kakaocdn.net/kakao_arena/shopping/train.chunk.03?TWGServiceId=kakao_arena&Expires=1615302245&Signature=gH6sgSeCZ0AbIHTB8gpmlx0QhxE%3D&AllowedIp=110.9.51.148&download' -O train.chunk.03

echo 'TrainData_04 Download'
wget 'http://arena.kakaocdn.net/kakao_arena/shopping/train.chunk.04?TWGServiceId=kakao_arena&Expires=1615302264&Signature=rQL7s1/e2KS8RzMjg5luP5XBj%2BE%3D&AllowedIp=110.9.51.148&download' -O train.chunk.04

echo 'TrainData_05 Download'
wget 'http://arena.kakaocdn.net/kakao_arena/shopping/train.chunk.05?TWGServiceId=kakao_arena&Expires=1615302277&Signature=exnm%2BFwCvkmLi3uHYogksPfmhNY%3D&AllowedIp=110.9.51.148&download' -O train.chunk.05

echo 'TrainData_06 Download'
wget 'http://arena.kakaocdn.net/kakao_arena/shopping/train.chunk.05?TWGServiceId=kakao_arena&Expires=1615302277&Signature=exnm%2BFwCvkmLi3uHYogksPfmhNY%3D&AllowedIp=110.9.51.148&download' -O train.chunk.06

echo 'TrainData_07 Download'
wget 'http://arena.kakaocdn.net/kakao_arena/shopping/train.chunk.07?TWGServiceId=kakao_arena&Expires=1615302336&Signature=cWxM%2BMe7yIllG9/By6QelVGbbr4%3D&AllowedIp=110.9.51.148&download' -O train.chunk.07

echo 'TrainData_08 Download'
wget 'http://arena.kakaocdn.net/kakao_arena/shopping/train.chunk.08?TWGServiceId=kakao_arena&Expires=1615302348&Signature=lXbe%2BGORp0h8%2BcF90ORaYaVqEec%3D&AllowedIp=110.9.51.148&download' -O train.chunk.08

echo 'TrainData_09 Download'
wget 'http://arena.kakaocdn.net/kakao_arena/shopping/train.chunk.09?TWGServiceId=kakao_arena&Expires=1615302361&Signature=6zX5KcHFFacqrxWbUxbS%2BMuBigM%3D&AllowedIp=110.9.51.148&download' -O train.chunk.09

echo 'TestData_02 Download'
wget 'http://arena.kakaocdn.net/kakao_arena/shopping/test.chunk.02?TWGServiceId=kakao_arena&Expires=1615302393&Signature=9ujJvwHLrbSQHliChr8R0Aqku0I%3D&AllowedIp=110.9.51.148&download'  -O test.chunk.02
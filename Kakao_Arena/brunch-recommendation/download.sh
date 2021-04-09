#!/bin/zsh

mkdir res
cd ./res

echo $PWD

echo 'Magazine.json Download'
wget 'https://arena.kakaocdn.net/kakao_arena/brunch_article_prediction/res/magazine.json?TWGServiceId=kakao_arena&Expires=1617971462&Signature=rmBDNam/GfCmv5Y2L8Tc1VvzG4w%3D&AllowedIp=110.9.51.148&download' -O magazine.json

echo 'Metadata.json Download'
wget 'https://arena.kakaocdn.net/kakao_arena/brunch_article_prediction/res/metadata.json?TWGServiceId=kakao_arena&Expires=1617971480&Signature=JhSH7KevJsfOqtK2etYcHMrGCk4%3D&AllowedIp=110.9.51.148&download' -O metadata.json

echo 'Users.json Download'
wget 'https://arena.kakaocdn.net/kakao_arena/brunch_article_prediction/res/users.json?TWGServiceId=kakao_arena&Expires=1617971507&Signature=tkhD6pnigmXnOT4bpZKhpkLBCBo%3D&AllowedIp=110.9.51.148&download' -O users.json

mkdir contents
cd ./contents
echo 'Contents File Download'

wget 'https://arena.kakaocdn.net/kakao_arena/brunch_article_prediction/res/contents/data.0?TWGServiceId=kakao_arena&Expires=1617971804&Signature=%2BVD10z0OjBywoJwbPo1vz8yiF5w%3D&AllowedIp=110.9.51.148&download' -O data.0

wget 'https://arena.kakaocdn.net/kakao_arena/brunch_article_prediction/res/contents/data.1?TWGServiceId=kakao_arena&Expires=1617971830&Signature=uDsXR7w9LqoVDnZaw6iyO1QBlq0%3D&AllowedIp=110.9.51.148&download' -O data.1

wget 'https://arena.kakaocdn.net/kakao_arena/brunch_article_prediction/res/contents/data.2?TWGServiceId=kakao_arena&Expires=1617971846&Signature=QbbTnoLDvLbsJgBM%2B3bfejqy8KU%3D&AllowedIp=110.9.51.148&download' -O data.2

wget 'https://arena.kakaocdn.net/kakao_arena/brunch_article_prediction/res/contents/data.3?TWGServiceId=kakao_arena&Expires=1617971859&Signature=Tme9XW1kDENEvwq9aQX0cQRjMN4%3D&AllowedIp=110.9.51.148&download' -O data.3

wget 'https://arena.kakaocdn.net/kakao_arena/brunch_article_prediction/res/contents/data.4?TWGServiceId=kakao_arena&Expires=1617971880&Signature=DIzOCDTYxOvDVNJMr3fjPJfZwgE%3D&AllowedIp=110.9.51.148&download' -O data.4

wget 'https://arena.kakaocdn.net/kakao_arena/brunch_article_prediction/res/contents/data.5?TWGServiceId=kakao_arena&Expires=1617971895&Signature=a7umAKJ/nrTHskkcxdQM19fKx5g%3D&AllowedIp=110.9.51.148&download' -O data.5

wget 'https://arena.kakaocdn.net/kakao_arena/brunch_article_prediction/res/contents/data.6?TWGServiceId=kakao_arena&Expires=1617971911&Signature=pHvkMsjwK2Rw6afItcIr4A/zoYM%3D&AllowedIp=110.9.51.148&download' -O data.6

cd ..

echo 'Predict.tar Download'
wget 'https://arena.kakaocdn.net/kakao_arena/brunch_article_prediction/res/predict.tar?TWGServiceId=kakao_arena&Expires=1617971556&Signature=8H8sizNBQAL6kd2udtknRf48wT0%3D&AllowedIp=110.9.51.148&download' -O predict.tar 

echo 'Unzip predcit.tar'
tar -xvf predict.tar

echo 'Read.tar Download'
wget 'https://arena.kakaocdn.net/kakao_arena/brunch_article_prediction/res/read.tar?TWGServiceId=kakao_arena&Expires=1617971679&Signature=BWUuDNNDaHRF29///AZd5j%2BgM4c%3D&AllowedIp=110.9.51.148&download'  -O read.tar

echo 'Unzip read.tar'
tar -xvf read.tar

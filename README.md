# 팀: 전전긍긍
예스아이캔(deactivate-Noisecancling) 개발, 노캔을 줄여서 이를 반대하는 의미임

## 개발 기간
- 24.09.20 ~ 24.10.08 (중간 보고)
- 24.11.01 ~ 24.12.06 (최종 보고)

## 맴버 구성
- 핵심 SW 개발 : 김지민
- 데이터셋 제공 : 박현준, 장현서
- 개요 및 보고서 : 박현준, 김지민
- 주요 발표 내용 : 장현서

## 개발 환경
- 파이썬 가상환경(3.8.10)
- Tensorflow 2.10.0
- Tensorflow-hub 0.12.0
- Numpy 1.23.5
- h5py 3.1.0
- librosa 0.8.0
- keras 2.10.0
- typing-extensions 3.7.4.3
- numba 0.53.1
- Ipython 7.16.1 
- flask 2.0.1
- werkzeug 2.0.3
- soundfile, matplotlib, scipy, pyaudio

## 사용 모델
- YAMnet
- 테스트 정확도: 0.9744

## 프로젝트 소개
무선 이어폰의 주요 기능중 노이즈캔슬링으로 인한 보행자 사고가 발생하고 있다.
이로 인한 사고를 줄이기 위해 능동적인 노이즈캔슬링 비활성화 기술을 제안한다.

## 주요 기능
자동차와 같은 위험 소리가 감지되면 자동으로 사용자의 노이즈캔슬링 기능을 일시적으로 비활성화한다.


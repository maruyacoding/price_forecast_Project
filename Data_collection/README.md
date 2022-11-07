DataFrame 통합 
---
Dacon data 활용 https://dacon.io/competitions/official/235801/overview/description
- raw data는 1달 단위로 존재
- Dacon data 2016.01 ~ 2019.09 data 통합
-> 프로젝트 데이터로 활용

1. 무, 배추, 대파, 건고추, 마늘, 양파, 사과 7가지 품종만 데이터 추출
2. 메모리 최소화를 위해 object -> category 타입으로 변경
3. 결측치 '.'으로 처리
4. SALEDATE(판매일) 데이터 타입 object -> datatime으로 변경
5, 거래가 취소(음수인 경우)나 0인 경우 drop -> 거래가 양수인 경우의 데이터만 진행
6. index 정렬
7. csv -> parquet 파일로 저장 (용량 1GB -> 159MB)

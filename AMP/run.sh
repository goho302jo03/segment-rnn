#! /bin/bash
# mkdir ./log/full
# for i in {1..10};
# do
# 	python3 main.py $i full > ./log/full/$i.log
# done
#
# mkdir -p ./log/resample_lstm/segment-1
# mkdir -p  ./log/non_resample_lstm/segment-1
for i in {1..10};
do
	for j in {10..150..10};
	do
		python3 main.py $i segment 1 $j > ./log/resample_lstm/segment-1/$j.$i.log
		# python3 main.py $i segment 0 $j 1 > ./log/non_resample_lstm/segment-1/$j.$i.log
	done
done


# mkdir ./log/segment-2-0
# for i in {1..10};
# do
# 	for j in {10..130..10};
# 	do
# 		python3 main.py segment 2 $j 0 > ./log/segment-2-0/$j.$i.log
# 	done
# done
#
# mkdir ./log/segment-3-0
# for i in {1..10};
# do
# 	for j in {10..100..10};
# 	do
# 		python3 main.py segment 3 $j 0 > ./log/segment-3-0/$j.$i.log
# 	done
# done
#
# mkdir ./log/segment-4-0
# for i in {1..10};
# do
# 	for j in {10..80..10};
# 	do
# 		python3 main.py segment 4 $j 0 > ./log/segment-4-0/$j.$i.log
# 	done
# done
#
# mkdir ./log/segment-5-0
# for i in {1..10};
# do
# 	for j in {10..60..10};
# 	do
# 		python3 main.py segment 5 $j 0 > ./log/segment-5-0/$j.$i.log
# 	done
# done

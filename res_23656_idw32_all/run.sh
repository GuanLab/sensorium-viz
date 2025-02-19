
# for num in 0 800 2000 5000 10000 20000; do
#     for img in ${num}_synthetic/*png; do
#         CUDA_VISIBLE_DEVICES=3 python metrics.py $img &
#     done
# done
# wait

# for num in 40000 80000; do
#     for img in ${num}_synthetic*/*png; do
#         CUDA_VISIBLE_DEVICES=3 python metrics_full.py $img &
#     done
# done
# wait

for num in 40000 80000; do
    for img in ${num}_synthetic*/*png; do
        CUDA_VISIBLE_DEVICES=0 python metrics_full.py $img &
    done
done
wait
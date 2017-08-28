if [ "$#" -eq 2 ]; then
    store_dir=$1
    data_yaml=$2
    dataset=`fgrep dataset $data_yaml | sed -e 's/dataset: //' | sed -e 's/datasets\///'`
    rm -f datasets/$dataset.data
    rm -f datasets/$dataset.idx
    rm -f $store_dir/$dataset.data
    rm -f $store_dir/$dataset.idx
    touch $store_dir/$dataset.data
    touch $store_dir/$dataset.idx
    ln -s $store_dir/$dataset.data datasets/$dataset.data
    ln -s $store_dir/$dataset.idx datasets/$dataset.idx
    echo Generating $data_yaml 
    invoke gen_train_dataset --config=$data_yaml &> $store_dir/$dataset.out
    exit
fi

if [ "$#" -ne 1 ]; then
    echo "gen_all.bash secondary-storage-dir"
    exit
fi
store_dir=$1

for data_yaml in gen_dataset/*.yaml
do
    bash gen_all.bash $store_dir $data_yaml &
    sleep 3
done

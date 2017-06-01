if [ "$#" -eq 2 ]; then
    s=$1
    t=$2
elif [ "$#" -eq 1 ]; then
        s=$1
        t=$1
elif [ "$#" -ne 1 ]; then
    echo "publish frames_dir [publish_name]"
    exit
fi

mkdir ~/public_html/$t
cp $s/test* ~/public_html/$t
album ~/public_html/$t

if [[ "$#" -ne 2 ]]; then
    echo "$0: frames_dir html_dir"
    exit
fi

x=$1
y=$2
mkdir ~/public_html/$y; cp $x/test* ~/public_html/$y; album ~/public_html/$y
cp $x/chart.png ~/public_html/$y

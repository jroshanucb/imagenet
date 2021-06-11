for filename in *.tar; do
    dirname=$(echo "$filename" | cut -f 1 -d '.')
    mkdir "$dirname" && mv "$filename" "$dirname" && cd "$dirname" && tar -xvf "$filename" && rm "$filename" && cd .. 
done

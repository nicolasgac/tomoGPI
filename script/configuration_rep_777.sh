sudo chown -R gac:$1 $2
chmod 2777 $2/

setfacl -Rdm g:$1:rwx $2

cd $2
find . -type f -exec chmod g+w {} \;
find . -type f -exec chmod a+rwx {} \;
find . -type d -exec chmod 2777 {} \;

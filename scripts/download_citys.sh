#!/bin/bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=yourID&password=yourpw&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=33
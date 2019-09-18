import requests
import contextlib
import sys
def download(url, session_id, save_path):
    cookies = {
        'PHPSESSID': session_id
    }
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Cookie': 'PHPSESSID=%s'%session_id,
        'DNT': '1',
        'Host': 'www.cityscapes-dataset.com',
        'Referer': 'https://www.cityscapes-dataset.com/downloads/',
        'Upgrade-Insecure-Request': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36'
    }
    res = requests.get(url, headers=headers, cookies=cookies, stream=True)
    with contextlib.closing(res) as r:
        accepts = 0
        with open(save_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)
                    accepts += len(chunk)
                    progress = accepts / int(r.headers['Content-Length'])
                    if accepts % 1000 ==0:
                        sys.stdout.write(("%.3f\n" % progress))
# download(
#     url='https://www.cityscapes-dataset.com/file-handling/?packageID=29',  
#     session_id='kuh7fe9nfkd9gp3785gt4m5e80',   
#     save_path='trainvaltest_foggy.zip'
# )

download(
    url='https://www.cityscapes-dataset.com/file-handling/?packageID=33',  
    session_id='kuh7fe9nfkd9gp3785gt4m5e80',   
    save_path='trainval_rain.zip'
)
# 参考：https://github.com/takurooo/Python-ImageNet_Downloader

#-------------------------------------------
# import
#-------------------------------------------
import sys
import os
import codecs
import collections
from urllib import request
import glob
from PIL import Image

#-------------------------------------------
# global
#-------------------------------------------


#-------------------------------------------
# functions
#-------------------------------------------
class ImageNetDownloader(object):
    WNID_CHILDREN_URL="http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid={}&full={}"
    WNID_TO_WORDS_URL="http://www.image-net.org/api/text/wordnet.synset.getwords?wnid={}"
    IMG_LIST_URL="http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid={}"
    #BBOX_URL="http://www.image-net.org/api/download/imagenet.bbox.synset?wnid={}"

    def __init__(self, root=None):
        #self.root = root or os.getcwd()
        #self.img_dir = os.path.join(self.root, 'img')
        #self.list_dir = os.path.join(self.root, 'list')
        #self.wnid = ""
        #os.makedirs(self.root, exist_ok=True)
        #os.makedirs(self.img_dir, exist_ok=True)
        #os.makedirs(self.list_dir, exist_ok=True)
        pass

    def _check_data(self, data):
        """
        リクエストで取得したデータが不正なデータかどうかチェックする
        間違ったwnidを指定した場合、b'Invalid url!'が返ってくる
        """
        INVALID_DATA = b'Invalid url!'
        assert data != INVALID_DATA, "Invalid wnid."

    def _check_wnid(self, wnid):
        """
        wnidの基本的な形式をチェックする
        wnidはnから始まりそのあとに9桁の数字が並ぶ
        数字の並びが有効な値かは確認が大変なのでチェックしない
        """
        assert wnid[0] == 'n', 'Invalid wnid : {}'.format(wnid)
        assert len(wnid) == 9, 'Invalid wnid : {}'.format(wnid)

    def _make_list(self, path):
        data_list = []
        with codecs.open(path, 'r', 'utf-8') as f:
            data_list = [line.rstrip() for line in f]
        return data_list

    def _make_imginfo(self, path):
        """
        fname url
        fname url
        .
        .
        のテキストファイルを
        {fname : url, fname : url, ....}
        の辞書形式に変換する
        """
        imginfo = collections.OrderedDict()
        for line in self._make_list(path):
            fname, url = line.rstrip().split(None, 1)
            imginfo[fname] = url
        return imginfo

    def _get_data_with_url(self, url, invalid_urls=None, timeout=10):
        try:
            with request.urlopen(url, timeout=timeout) as response:
                if invalid_urls is not None:
                    for invalid_url in invalid_urls:
                        if response.geturl() == invalid_url:
                            return None

                html = response.read() # binary -> str
        except:
            return None

        return html

    def _download_imglist(self, path, wnid):
        out_path = os.path.join(path, wnid+'.txt')

        data = self._get_data_with_url(self.IMG_LIST_URL.format(wnid))
        self._check_data(data)

        if data is not None:
            data = data.decode().split()
            # data = [fname_0, url_0, fname_1, url_1, .....]
            fnames = data[::2]
            urls = data[1::2]
            with codecs.open(out_path, 'w', 'utf-8') as f:
                for fname, url in zip(fnames, urls):
                    write_format = "{} {}\n".format(fname, url)
                    f.write(write_format)

    def _download_imgs(self, path, imginfo, limit=0, verbose=False, timeout=10):
        UNAVAILABLE_IMG_URL="https://s.yimg.com/pw/images/en-us/photo_unavailable.png"
        num_of_img = len(imginfo)
        n_saved = 0
        for i, (fname, url) in enumerate(imginfo.items()):
            if verbose:
                print('{:5}/{:5} fname: {}  url: {}'.format(i+1, num_of_img, fname, url))

            out_path = os.path.join(path, fname+'.jpg')
            if os.path.exists(out_path):
                continue

            invalid_urls = [UNAVAILABLE_IMG_URL]
            img = self._get_data_with_url(url, invalid_urls, timeout=timeout)

            if img is None:
                continue

            with open(out_path, 'wb') as f:
                f.write(img)
            n_saved += 1

            if verbose:
                print('\tsaved[{}] to {}'.format(n_saved, out_path))

            if limit != 0 and limit <= n_saved:
                break

    def wnid_children(self, wnid, recursive=False):
        """
        指定したwnidの下層のwnidを取得
        recursive=Trueで最下層まで探索
        """
        self._check_wnid(wnid)
        full = 0
        if recursive:
            full = 1
        data = self._get_data_with_url(self.WNID_CHILDREN_URL.format(wnid, full))
        self._check_data(data)

        children = data.decode().replace('-', '').split()
        return children # [parent, child, child, child....]

    def wnid_to_words(self, wnid):
        """
        wnidを対応するsynsetを取得
        """
        self._check_wnid(wnid)
        data = self._get_data_with_url(self.WNID_TO_WORDS_URL.format(wnid))
        self._check_data(data)

        words = data.decode().split('\n')
        words = [word for word in words if word]
        return words

    def download(self, wnid, img_dir, list_dir, limit=0, verbose=False, timeout=10):
        """
        指定したWNIDに属する画像をimgフォルダに保存

        Parameters
        --------------
        wnid : str
            ダウンロードする画像のWNID
        image_dir : str
            画像を保存するディレクトリ名
        list_dir : str
            読み込んだデータ一覧を格納するディレクトリ
            指定したディレクトリ内に「<wnid>.txt」のファイルが作成される
            （※デフォルトでは一覧を作成しないようにしたい）
        limit : int
            ダウンロード画像の上限
        verbose : bool
            ログを表示するかどうか
        timeout : float
            各URLへの通信のタイムアウト時間を秒数で指定
        """
        self.wnid = ""
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(list_dir, exist_ok=True)

        self._check_wnid(wnid)

        list_path = os.path.join(list_dir, wnid+'.txt')
        if not os.path.exists(list_path):
            self._download_imglist(list_dir, wnid)
        imginfo = self._make_imginfo(list_path)

        os.makedirs(img_dir, exist_ok=True)

        self._download_imgs(img_dir, imginfo, limit, verbose, timeout=timeout)


def delete_unreadable_image(dir_path, extension='jpg'):
    '''
    PILで読み込めない画像データを削除する
    
    Parameters
    ---------------
    dir_path : str
        調べたいディレクトリへのパス
        このディレクトリ内の指定した拡張子のファイル全て調べる
    extension : str
        調べるファイルの拡張子
    '''
    # 指定した拡張子のファイルリストを作成
    img_path_list = glob.glob(os.path.join(dir_path, '**/*.'+extension))
    
    # 全画像を調べる
    for path in img_path_list:
        try:
            img = Image.open(path)
        except:
            os.remove(path)
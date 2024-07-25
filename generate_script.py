import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from google.colab import auth
auth.authenticate_user()
from google.cloud import aiplatform
aiplatform.init(project="herbsvideoscriptgenerator")
def generate():
    vertexai.init(project="herbsvideoscriptgenerator", location="us-central1")
    model = GenerativeModel("gemini-1.5-flash-001")
    responses = model.generate_content(
        [text1],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )
    script_output = ""
    for response in responses:
        script_output += response.text
    with open('/content/script/天麻.txt', 'w', encoding='utf-8') as f:
        f.write(script_output)
text1 = """根据提供给你的药材简介，模仿以下文本用中文写两个药材script，要以让人听懂，引人入胜的方式，用作于抖音分享中国药材的script
要写文本和其对应的画面，一句一行 eg.（（画面展示金银花藤蔓，开满白色小花），每个素材大概15行文本，15行画面
这是我需要写的药材：天麻
素材1：参考例子1，例子2，例子3。需要写出该药材的样子，制作过程，功效
素材2：参考例子4。细节该药材的历史故事，大概两分钟
例子1：
今天我们来认识一种植物
地黄有的地方叫它生地
小即喝酒
地黄为玄生科地黄属
多年生草本植物
如要不为是它的根块
地黄同一种中草
要可以产生两种药材
直接挖出的先地黄
和经过炮制的属地黄
价值完全不同
所以有生地和属地的区分
看看您有这种植物吗

例子2：
大家看這個開子色小花的植物你們認識嗎
它就是帝皇是一種極為珍貴的草本
帝皇是學生科書的一種波年生植物
它全租上下有容毛
裂片成土原形邊緣有巨蜘
它的花像一個小纜吧
記得小時候還經常摘它來吸一吸
像風蜜一樣甜
它的根金為黃白色
因此而得名
在過去作為展廖來使用
帝皇作為一種古老的植物
在我國有著悠久的歷史
旗下值從周朝開始就被人們發現
在當時並被作為
公平為皇室所用
所以帝皇不僅僅是一位珍貴的草本植物
更是一種文化的傳承
只可惜野生的帝皇越來越赏了
你的家鄉有帝皇吗
一起來說說吧

例子3：你别看他不起眼了
实际上他有非常高的药用价值
我奶奶撑他为黄洋子根
其实我们这边很多人叫他秘密棍
这种开着紫色小花的
就是我们大名顶顶的地黄
比如打个比方
人家都坐着飞机跑了几圈
你还在厕所蹲着的时候
你也可以挂他的根来用
他主要分不了河南山东一带
地黄的花朵里面有很多字业
我们小时候常常把它揪掉
寒在嘴里
西他的蜜
甜甜的像风蜜一样
小时候你有没有做过这样的事情的
在生活中不小心被文章盯鸟
也有很多人拿他的根来涂抹
在很久以前古书里就有记载
地黄是一种不周不扣的很明贵的重要
懂他的人很喜欢拿他的根来包州
有很多人挖这种野生的地黄来送卖
他的价格高达六七十块钱一斤
他不仅可以两半
还可以泡酒喝
六位地黄完中就有这种地黄

例子4：
我给大家讲一个黄金的传说
这传说很久的时候
大概也就是三国时期划脱时代吧
有这么一个压换
不堪忍受
咱说叫地主老财的那种压迫吧
领主
然后只身就跑到了身上老林里边
当时的压换的是受小哭干
咱说提弱多病
那么大家想一下
跑到身上老林里边
不实
也得为野兽持了
但是谁成想三年以后
大家火
偶然在山里面发现这个压换以后发现什么
这个压换不仅没使
而且身体健庄
啊这个荣贸非常非常的健康
脸色非常红润
所以大家火就问你吃了什么东西
所以压换就讲我吃了一种不明
是吧
明子的炒药的根筋
后来这事儿被谁听到呢
当时的大一家华脱
说华脱听到了以后就赶紧来
找来这种炒药研究
那么跑了他这个根细以后发现
他就跟特别像什么呢
像这个鸡头
特别像鸡头
所以他也叫鸡头身嘛
然后他就回去给老火就做思安嘛
这就是黄金的传说
所以黄金是咱们中药里边
别名最多的一个
齐剧
药用价值和营养价值一种炒药
那么他还有什么别名
大家火可以在评论区里边留言
"""
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}
generate()
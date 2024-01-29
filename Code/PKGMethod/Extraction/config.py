# SeqGPT_model_path = '/mnt/b20f528c-613a-48b9-a633-d0b37eb76f7b/PKG2/SeqGPT/nlp_seqgpt-560m'
Zephyr_7B_model_path = '/mnt/LLMs/zephyr-7b-beta'

out_schema_result_path = 'extra.log'


# NER_schema = ['专业名称', '数字', '人名', '性别', '国家', '电子邮件', '政治组织', '职位', '菜品', '作品', '景点', '时间', '其他', '组织', '地址', '影视']
NER_schema = ['专业名称', '生理特征', '人名', '性别', '国籍', '电子邮件', '政治组织', '职务', '菜品', '音乐类型', '景点', '时间', '组织', '地址', '影视', '身份证号', '银行卡号', '电话号码', '国籍', '金额', '账号', '用户名', '其他']

RE_attribute_schema_level_1 = ['基本属性', '联系方式', '住址', '出生', '学历背景', '政治面貌', '生理特征', '性格类型', '工作背景', '婚姻情况', '个人偏好']
RE_relation_schema_level_1 = ['同学', '同事', '合作伙伴', '家人', '婚亲关系', '法定关系', '恋人', '朋友', '教师', '学生']

DuIE_relation_schema = ['毕业院校', '嘉宾', '主题曲', '代言人', '所属专辑', '父亲', '作者', '母亲', '专业代码', '占地面积', '邮政编码', '注册资本', '主角', '妻子', '编剧', '气候', '歌手', '校长', '创始人', '首都', '丈夫', '朝代', '面积', '总部地点', '祖籍', '人口数量', '制片人', '修业年限', '所在城市', '董事长', '作词', '改编自', '出品公司', '导演', '作曲', '主演', '主持人', '成立日期', '简称', '海拔', '号', '国籍', '官方语言']

# RE attr 11-1
RE_attribute_schema = ['昵称', '性别', '年龄', '身份证号', '银行卡号', '国籍',
                       '电话号码', '电子邮箱',
                       '家庭地址', '公司地址',
                       '出生地', '生日',
                       '学历', '学校名称', '入学时间', '毕业时间', '专业名称',
                       '政治面貌',
                       '身高', '体重',
                       '性格类型',
                       '工作开始时间', '工作结束时间', '工作单位', '工作职务', '工作薪资',
                       '婚姻情况',
                       '电影偏好', '菜品偏好', '音乐偏好', '购物偏好', '景点偏好', '酒店偏好']
RE_attribute_schema_dict = {'姓名': '基本属性/姓名', '昵称': '基本属性/昵称', '性别': '基本属性/性别', '年龄': '基本属性/年龄', '身份证号': '基本属性/身份证号', '银行卡号': '基本属性/银行卡号', '国籍': '基本属性/国籍', '电话号码': '联系方式/电话号码', '电子邮箱': '联系方式/电子邮箱', '家庭地址': '住址/家庭地址', '公司地址': '住址/公司地址', '出生地': '出生/出生地', '生日': '出生/生日', '学历': '教育背景/学历', '学校名称': '教育背景/学校名称', '入学时间': '教育背景/入学时间', '毕业时间': '教育背景/毕业时间', '专业名称': '教育背景/专业名称', '政治面貌': '政治面貌', '身高': '生理特征/身高', '体重': '生理特征/体重', '性格类型': '性格类型', '工作开始时间': '工作背景/工作开始时间', '工作结束时间': '工作背景/工作结束时间', '工作单位': '工作背景/工作单位', '工作职务': '工作背景/工作职务', '工作薪资': '工作背景/工作薪资', '婚姻情况': '婚姻情况', '电影偏好': '个人偏好/电影偏好', '菜品偏好': '个人偏好/菜品偏好', '音乐偏好': '个人偏好/音乐偏好', '购物偏好': '个人偏好/购物偏好', '景点偏好': '个人偏好/景点偏好', '酒店偏好': '个人偏好/酒店偏好'}
RE_attribute_type_schema_dict = {'基本属性/姓名': '人名', '基本属性/昵称': '人名', '基本属性/性别': '性别', '基本属性/年龄': '年龄', '基本属性/身份证号': '身份证号', '基本属性/银行卡号': '银行卡号', '基本属性/国籍': '国籍', '联系方式/电话号码': '电话号码', '联系方式/电子邮箱': '电子邮件', '住址/家庭地址': '地址', '住址/公司地址': '地址', '出生/出生地': '地址', '出生/生日': '时间', '教育背景/学历': '其他', '教育背景/学校名称': '组织', '教育背景/入学时间': '时间', '教育背景/毕业时间': '时间', '教育背景/专业名称': '专业名称', '政治面貌': '政治组织', '生理特征/身高': '生理特征', '生理特征/体重': '生理特征', '性格类型': '其他', '工作背景/工作开始时间': '时间', '工作背景/工作结束时间': '时间', '工作背景/工作单位': '组织', '工作背景/工作职务': '职务', '工作背景/工作薪资': '金额', '婚姻情况': '其他', '个人偏好/电影偏好': '影视', '个人偏好/菜品偏好': '菜品', '个人偏好/音乐偏好': '音乐类型', '个人偏好/购物偏好': '其他', '个人偏好/景点偏好': '景点', '个人偏好/酒店偏好': '组织'}
RE_relation_schema = ['幼儿园同学', '小学同学', '初中同学', '高中同学', '大学同学', '研究生同学', '前同事', '上级同事', '平级同事', '下级同事', '甲方/乙方领导', '客户', '顾问', '中介', '子女老师', '合伙人', '房东', '租客', '父亲', '母亲', '兄弟', '姊妹', '儿子', '女儿', '配偶', '旁系亲属', '爷爷', '奶奶', '外公', '外婆', '岳父', '岳母', '公公', '婆婆', '亲家', '小姨子', '小叔子', '儿媳', '女婿', '监护人', '被监护人', '继承人', '被继承人', '养父', '养母', '养子', '养女', '继父', '继母', '继子', '继女', '前恋人', '现恋人', '密友', '好友', '日常朋友', '熟人', '距离朋友', '网友', '学校老师', '辅导班老师', '大学导师', '前学生', '现学生']
RE_relation_schema_dict = {'幼儿园同学': '同学/幼儿园同学', '小学同学': '同学/小学同学', '初中同学': '同学/初中同学', '高中同学': '同学/高中同学', '大学同学': '同学/大学同学', '研究生同学': '同学/研究生同学', '前同事': '同事/前同事', '上级同事': '同事/上级同事', '平级同事': '同事/平级同事', '下级同事': '同事/下级同事', '甲方/乙方领导': '合作伙伴/甲方/乙方领导', '客户': '合作伙伴/客户', '顾问': '合作伙伴/顾问', '中介': '合作伙伴/中介', '子女老师': '合作伙伴/子女老师', '合伙人': '合作伙伴/合伙人', '房东': '合作伙伴/房东', '租客': '合作伙伴/租客', '父亲': '家人/父亲', '母亲': '家人/母亲', '兄弟': '家人/兄弟', '姊妹': '家人/姊妹', '儿子': '家人/儿子', '女儿': '家人/女儿', '配偶': '家人/配偶', '旁系亲属': '家人/旁系亲属', '爷爷': '家人/爷爷', '奶奶': '家人/奶奶', '外公': '家人/外公', '外婆': '家人/外婆', '岳父': '婚亲关系/岳父', '岳母': '婚亲关系/岳母', '公公': '婚亲关系/公公', '婆婆': '婚亲关系/婆婆', '亲家': '婚亲关系/亲家', '小姨子': '婚亲关系/小姨子', '小叔子': '婚亲关系/小叔子', '儿媳': '婚亲关系/儿媳', '女婿': '婚亲关系/女婿', '监护人': '法定关系/监护人', '被监护人': '法定关系/被监护人', '继承人': '法定关系/继承人', '被继承人': '法定关系/被继承人', '养父': '法定关系/养父', '养母': '法定关系/养母', '养子': '法定关系/养子', '养女': '法定关系/养女', '继父': '法定关系/继父', '继母': '法定关系/继母', '继子': '法定关系/继子', '继女': '法定关系/继女', '前恋人': '恋人/前恋人', '现恋人': '恋人/现恋人', '密友': '朋友/密友', '好友': '朋友/好友', '日常朋友': '朋友/日常朋友', '熟人': '朋友/熟人', '距离朋友': '朋友/距离朋友', '网友': '朋友/网友', '学校老师': '教师/学校老师', '辅导班老师': '教师/辅导班老师', '大学导师': '教师/大学导师', '前学生': '学生/前学生', '现学生': '学生/现学生'}

# EE_schema_classification = \
#     '美食, 休闲娱乐, 演出电影, 工作/出差, 工作/会议, 工作/团建, 工作/培训, 工作/应酬, 工作/邮件, 工作/项目, ' \
#     '汽车/买车, 汽车/修车, 汽车/保险, 汽车/加油, 汽车/4S店, 汽车/洗车, 汽车/年检, 汽车/汽车美容, 汽车/罚单, ' \
#     '购物, 运动健身, 学习培训, 医疗/中医, 医疗/看望病人, 医疗/住院, 医疗/手术, 医疗/体检, 医疗/买药, 医疗/看病, ' \
#     '生活/存钱, 生活/取钱, 生活/买票, 生活/洗衣, 生活/家政, 生活/维修, 生活/房产, 生活/政府, 生活/缴费, 生活/影印, ' \
#     '生活/约会, 生活/快递, 生活/旅游, 生活/结婚, 生活/外出游玩, 生活/酒店, 出行'
# EE_schema_dict = {'美食': ['地点', '价格', '预约时间', '参与者'], '休闲娱乐': ['地点', '时长', '价格', '预约时间', '参与者'], '演出电影': ['地点', '主题', '时长', '价格', '预约时间', '参与者', '电影类型', '电影名'], '工作/出差': ['出发地', '目的地', '出发时间', '到达时间', '工作内容', '费用'], '工作/会议': ['地点', '参与者', '时间', '会议名称', '负责人', '角色'], '工作/团建': ['地点', '参与者', '时间', '主题'], '工作/培训': ['地点', '参与者', '时间', '主题'], '工作/应酬': ['地点', '参与者', '时间'], '工作/邮件': ['发件人', '收件人', '简要内容', '发送时间'], '工作/项目': ['名称', '描述', '开始日期', '结束日期', '负责人'], '汽车/买车': ['品牌', '车型', '价格', '购买日期', '车牌号', '经销商'], '汽车/修车': ['车牌号', '修理内容', '修理费用', '修理日期', '修理地点'], '汽车/保险': ['保险类型', '保险公司', '保费', '购买日期', '到期日期'], '汽车/加油': ['加油站', '油品类型', '加油量', '加油日期', '费用'], '汽车/4S店': ['服务类型', '名称', '日期', '费用', '时间'], '汽车/洗车': ['洗车类型', '地点', '日期', '费用', '时间'], '汽车/年检': ['地点', '日期', '费用', '年检结果', '下次年检日期'], '汽车/汽车美容': ['美容类型', '地点', '日期', '费用'], '汽车/罚单': ['违规类型', '地点', '发出者', '接收者', '发出日期', '金额'], '购物': ['地点', '日期', '花费', '商品种类', '购买/未购买'], '运动健身': ['地点', '日期', '费用', '时长', '教练', '参与者'], '学习培训': ['语种', '授课机构', '费用', '日期', '授课时长', '参与者', '画种', '乐器', '舞种', '字体', '学科', '类型'], '医疗/中医': ['病症', '医院', '费用', '日期', '中药方', '医生'], '医疗/看望病人': ['病人姓名', '医院', '日期', '礼物', '病人情况'], '医疗/住院': ['病症', '医院', '费用', '日期', '出院日期', '房间类型'], '医疗/手术': ['病症', '医院', '费用', '日期', '手术时长', '医生'], '医疗/体检': ['体检项目', '医院', '费用', '日期', '体检结果'], '医疗/买药': ['药品名称', '药店', '费用', '日期', '剂量'], '医疗/看病': ['病症', '医院', '费用', '日期', '处方药', '医生'], '生活/存钱': ['存款金额', '银行', '费用', '日期', '存款时长', '利率'], '生活/取钱': ['取款金额', '银行', '费用', '日期', '卡类型'], '生活/买票': ['票种', '购票处', '费用', '日期'], '生活/洗衣': ['衣物种类', '洗衣店', '费用', '日期', '取衣日期', '洗衣方式'], '生活/家政': ['服务类型', '家政公司', '费用', '日期', '服务时长', '家政员'], '生活/维修': ['维修项目', '维修店', '费用', '日期', '预计完成', '维修员'], '生活/房产': ['地点', '中介/个人', '费用', '购买/租赁', '面积', '位置'], '生活/政府': ['服务类型', '政府机构', '费用', '日期', '处理时间', '工作人员'], '生活/缴费': ['费用类型', '收费机构', '费用', '日期', '缴费方式', '截止日期'], '生活/影印': ['文档名', '影印店', '费用', '日期', '页数'], '生活/约会': ['地点', '日期', '时长', '参与者', '活动'], '生活/快递': ['包裹类型', '快递公司', '费用', '日期', '预计到达', '快递员', '收货地址'], '生活/旅游': ['目的地', '旅行社/自助', '费用', '日期', '返回日期', '住宿', '行程', '参与者'], '生活/结婚': ['地点', '日期', '费用', '新郎', '新娘', '参与者'], '生活/外出游玩': ['地点', '日期', '费用', '活动类型', '参与者'], '生活/酒店': ['酒店名称', '预订平台', '费用', '入住日期', '退房日期', '房型'], '出行': ['出发地', '目的地', '出发时间', '到达时间', '班次', '同行人', '车牌号', '班次/车牌号', '延误时间']}
# EE_schema_dict = {'美食': ['地点', '价格', '预约时间', '参与者'], '休闲娱乐': ['地点', '时长', '价格', '预约时间', '参与者'], '演出电影': ['地点', '主题', '时长', '价格', '预约时间', '参与者', '电影类型', '电影名'], '工作/出差': ['出发地', '目的地', '出发时间', '到达时间', '工作内容', '费用'], '工作/会议': ['地点', '参与者', '时间', '会议名称', '负责人', '角色'], '工作/团建': ['地点', '参与者', '时间', '主题'], '工作/培训': ['地点', '参与者', '时间', '主题'], '工作/应酬': ['地点', '参与者', '时间'], '工作/邮件': ['发件人', '收件人', '简要内容', '发送时间'], '工作/项目': ['名称', '描述', '开始时间', '结束时间', '负责人'], '汽车/买车': ['品牌', '车型', '价格', '购买时间', '车牌号', '经销商'], '汽车/修车': ['车牌号', '修理内容', '修理费用', '时间', '地点'], '汽车/保险': ['保险类型', '保险公司', '保费', '购买时间', '到期时间'], '汽车/4S店': ['服务类型', '时间', '地点', '费用'], '汽车/洗车': ['地点', '时间', '费用'], '汽车/年检': ['地点', '时间', '费用', '年检结果', '下次年检时间'], '汽车/汽车美容': ['美容类型', '地点', '时间', '费用'], '汽车/罚单': ['违规类型', '地点', '发出者', '接收者', '发出时间', '金额'], '购物': ['地点', '时间', '花费', '已购商品', '网购平台'], '运动健身': ['地点', '时间', '费用', '时长', '教练', '参与者'], '学习培训': ['语种', '授课机构', '费用', '时间', '地点', '授课时长', '参与者', '画种', '乐器', '舞种', '字体', '学科', '类型'], '医疗/中医': ['病症', '医院', '费用', '时间', '地点', '中药方', '医生'], '医疗/看望病人': ['病人姓名', '医院', '时间', '地点', '礼物', '病人情况'], '医疗/住院': ['病症', '医院', '费用', '时间', '地点', '出院时间', '房间类型'], '医疗/手术': ['病症', '医院', '费用', '时间', '地点', '手术时长', '医生'], '医疗/体检': ['体检项目', '医院', '费用', '时间', '地点', '体检结果'], '医疗/买药': ['药品名称', '药店', '费用', '时间', '地点', '剂量'], '医疗/看病': ['病症', '医院', '费用', '时间', '地点', '处方药', '医生'], '生活/存钱': ['存款金额', '银行', '费用', '时间', '存款时长', '利率'], '生活/取钱': ['取款金额', '银行', '费用', '时间', '银行卡类型'], '生活/洗衣': ['衣物种类', '洗衣店', '费用', '时间', '取衣时间', '洗衣方式'], '生活/家政': ['服务类型', '家政公司', '费用', '时间', '服务时长', '家政员'], '生活/维修': ['维修项目', '维修店', '费用', '时间', '地点', '预计完成', '维修员'], '生活/政府': ['服务类型', '政府机构', '费用', '处理时间', '工作人员'], '生活/缴费': ['费用类型', '收费机构', '费用', '缴费时间', '缴费方式', '截止时间'], '生活/影印': ['地点', '时间', '文档名', '影印店', '费用', '页数'], '生活/约会': ['地点', '时间', '参与者', '活动'], '生活/快递': ['快递单号', '快递公司', '费用', '实际到达时间', '预计到达时间', '快递员', '取件码', '收货地址'], '生活/旅游': ['地点', '出发时间', '旅行社', '费用', '返回时间', '参与者'], '生活/结婚': ['地点', '时间', '费用', '新郎', '新娘', '参与者'], '生活/外出游玩': ['地点', '时间', '费用', '活动类型', '参与者'], '生活/酒店': ['酒店名称', '预订平台', '费用', '入住时间', '退房时间', '房型'], '生活/租房': ['看房地点', '看房时间', '中介', '费用', '租赁开始时间', '租赁时长', '房屋面积', '房屋位置'], '出行': ['出发地', '目的地', '出发时间', '到达时间', '班次', '同行人', '车牌号', '班次/车牌号', '延误时间']}
# EE_schema_dict = {'美食/正餐': ['地点', '价格', '预约时间', '参与者'], '美食/地方菜': ['地点', '价格', '预约时间', '参与者'], '美食/火锅': ['地点', '价格', '预约时间', '参与者'], '美食/烧烤': ['地点', '价格', '预约时间', '参与者'], '美食/西餐': ['地点', '价格', '预约时间', '参与者'], '美食/外国菜': ['地点', '价格', '预约时间', '参与者'], '美食/下午茶': ['地点', '价格', '预约时间', '参与者'], '美食/甜点': ['地点', '价格', '预约时间', '参与者'], '美食/咖啡': ['地点', '价格', '预约时间', '参与者'], '美食/小吃': ['地点', '价格', '预约时间', '参与者'], '美食/宵夜': ['地点', '价格', '预约时间', '参与者'], '美食/农家乐': ['地点', '价格', '预约时间', '参与者'], '美食/家常菜': ['地点', '价格', '预约时间', '参与者'], '休闲娱乐/按摩': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/足疗': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/洗浴': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/汗蒸': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/唱歌': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/喝茶': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/棋牌': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/桌游': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/酒吧': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/密室逃脱': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/网吧': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/DIY手工': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/游乐': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/采摘': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/野餐': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/聚会': ['地点', '时长', '价格', '预约时间', '参与者'], '演出电影/演唱会': ['地点', '主题', '时长', '价格', '预约时间', '参与者'], '演出电影/音乐节': ['地点', '主题', '时长', '价格', '预约时间', '参与者'], '演出电影/音乐会': ['地点', '主题', '时长', '价格', '预约时间', '参与者'], '演出电影/歌剧': ['地点', '主题', '时长', '价格', '预约时间', '参与者'], '演出电影/话剧': ['地点', '主题', '时长', '价格', '预约时间', '参与者'], '演出电影/艺术活动': ['地点', '主题', '时长', '价格', '预约时间', '参与者'], '演出电影/表演': ['地点', '主题', '时长', '价格', '预约时间', '参与者'], '演出电影/电影': ['地点', '电影类型', '电影名', '时长', '价格', '预约时间', '参与者'], '工作/出差': ['出发地', '目的地', '出发时间', '到达时间', '工作内容', '费用'], '工作/会议': ['地点', '参与者', '时间', '会议名称', '负责人', '角色'], '工作/团建': ['地点', '参与者', '时间', '主题'], '工作/培训': ['地点', '参与者', '时间', '主题'], '工作/应酬': ['地点', '参与者', '时间'], '工作/邮件': ['发件人', '收件人', '简要内容', '发送时间'], '工作/项目': ['名称', '描述', '开始时间', '结束时间', '负责人'], '汽车/买车': ['品牌', '车型', '价格', '购买时间', '车牌号', '经销商'], '汽车/修车': ['车牌号', '修理内容', '修理费用', '时间', '地点'], '汽车/保险': ['保险类型', '保险公司', '保费', '购买时间', '到期时间'], '汽车/4S店': ['服务类型', '时间', '地点', '费用'], '汽车/洗车': ['地点', '时间', '费用'], '汽车/年检': ['地点', '时间', '费用', '年检结果', '下次年检时间'], '汽车/汽车美容': ['美容类型', '地点', '时间', '费用'], '汽车/罚单': ['违规类型', '地点', '发出者', '接收者', '发出时间', '金额'], '购物/逛街': ['地点', '时间', '花费', '已购商品'], '购物/网购': ['地点', '时间', '网购平台', '花费', '已购商品'], '运动健身/健身': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/飞盘': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/游泳': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/羽毛球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/溜冰': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/滑冰': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/射击': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/射箭': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/篮球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/桌球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/网球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/攀岩': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/乒乓球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/足球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/高尔夫球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/保龄球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/壁球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/跆拳道': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/柔道': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/泰拳': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/拳击': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/赛车': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/棒球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/冰球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/自行车': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/橄榄球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/马拉松': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/跑步': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/滑雪': ['地点', '时间', '费用', '时长', '教练', '参与者'], '学习培训/外语': ['语种', '授课机构', '费用', '时间', '地点', '授课时长', '参与者'], '学习培训/美术': ['画种', '授课机构', '费用', '时间', '地点', '授课时长', '参与者'], '学习培训/音乐': ['乐器', '授课机构', '费用', '时间', '地点', '授课时长', '参与者'], '学习培训/舞蹈': ['舞种', '授课机构', '费用', '时间', '地点', '授课时长', '参与者'], '学习培训/书法': ['字体', '授课机构', '费用', '时间', '地点', '授课时长', '参与者'], '学习培训/学科辅导': ['学科', '授课机构', '费用', '时间', '地点', '授课时长', '参与者'], '学习培训/驾校': ['类型', '授课机构', '费用', '时间', '地点', '授课时长', '参与者'], '医疗/中医': ['病症', '医院', '费用', '时间', '地点', '中药方', '医生'], '医疗/看望病人': ['病人姓名', '医院', '时间', '地点', '礼物', '病人情况'], '医疗/住院': ['病症', '医院', '费用', '时间', '地点', '出院时间', '房间类型'], '医疗/手术': ['病症', '医院', '费用', '时间', '地点', '手术时长', '医生'], '医疗/体检': ['体检项目', '医院', '费用', '时间', '地点', '体检结果'], '医疗/买药': ['药品名称', '药店', '费用', '时间', '地点', '剂量'], '医疗/看病': ['病症', '医院', '费用', '时间', '地点', '处方药', '医生'], '生活/存钱': ['存款金额', '银行', '费用', '时间', '存款时长', '利率'], '生活/取钱': ['取款金额', '银行', '费用', '时间', '银行卡类型'], '生活/洗衣': ['衣物种类', '洗衣店', '费用', '时间', '取衣时间', '洗衣方式'], '生活/家政': ['服务类型', '家政公司', '费用', '时间', '服务时长', '家政员'], '生活/维修': ['维修项目', '维修店', '费用', '时间', '地点', '预计完成', '维修员'], '生活/政府': ['服务类型', '政府机构', '费用', '处理时间', '工作人员'], '生活/缴费': ['费用类型', '收费机构', '费用', '缴费时间', '缴费方式', '截止时间'], '生活/影印': ['地点', '时间', '文档名', '影印店', '费用', '页数'], '生活/约会': ['地点', '时间', '参与者', '活动'], '生活/快递': ['快递单号', '快递公司', '费用', '实际到达时间', '预计到达时间', '快递员', '取件码', '收货地址'], '生活/旅游': ['地点', '出发时间', '旅行社', '费用', '返回时间', '参与者'], '生活/结婚': ['地点', '时间', '费用', '新郎', '新娘', '参与者'], '生活/外出游玩': ['地点', '时间', '费用', '活动类型', '参与者'], '生活/酒店': ['酒店名称', '预订平台', '费用', '入住时间', '退房时间', '房型'], '生活/租房': ['看房地点', '看房时间', '中介', '费用', '租赁开始时间', '租赁时长', '房屋面积', '房屋位置'], '出行/高铁/火车': ['出发地', '目的地', '出发时间', '到达时间', '班次', '同行人'], '出行/飞机': ['出发地', '目的地', '出发时间', '到达时间', '班次', '同行人'], '出行/网约车出行': ['出发地', '目的地', '出发时间', '到达时间', '车牌号', '同行人'], '出行/公共汽车': ['出发地', '目的地', '出发时间', '到达时间', '班次', '同行人'], '出行/私家车出行': ['出发地', '目的地', '出发时间', '到达时间', '车牌号', '同行人'], '出行/船渡': ['出发地', '目的地', '出发时间', '到达时间', '班次', '同行人'], '出行/延误': ['班次/车牌号', '延误时间']}

# 11-1
EE_schema_dict = {'其他': ['地点', '时间', '参与者'], '美食/正餐': ['地点', '价格', '预约时间', '参与者'], '美食/地方菜': ['地点', '价格', '预约时间', '参与者'], '美食/火锅': ['地点', '价格', '预约时间', '参与者'], '美食/烧烤': ['地点', '价格', '预约时间', '参与者'], '美食/西餐': ['地点', '价格', '预约时间', '参与者'], '美食/外国菜': ['地点', '价格', '预约时间', '参与者'], '美食/下午茶': ['地点', '价格', '预约时间', '参与者'], '美食/甜点': ['地点', '价格', '预约时间', '参与者'], '美食/咖啡': ['地点', '价格', '预约时间', '参与者'], '美食/小吃': ['地点', '价格', '预约时间', '参与者'], '美食/宵夜': ['地点', '价格', '预约时间', '参与者'], '美食/农家乐': ['地点', '价格', '预约时间', '参与者'], '美食/家常菜': ['地点', '价格', '预约时间', '参与者'], '休闲娱乐/按摩': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/足疗': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/洗浴': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/汗蒸': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/唱歌': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/喝茶': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/棋牌': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/桌游': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/酒吧': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/密室逃脱': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/网吧': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/DIY手工': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/游乐': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/采摘': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/野餐': ['地点', '时长', '价格', '预约时间', '参与者'], '休闲娱乐/聚会': ['地点', '时长', '价格', '预约时间', '参与者'], '演出电影/演唱会': ['地点', '主题', '时长', '价格', '预约时间', '参与者'], '演出电影/音乐节': ['地点', '主题', '时长', '价格', '预约时间', '参与者'], '演出电影/音乐会': ['地点', '主题', '时长', '价格', '预约时间', '参与者'], '演出电影/歌剧': ['地点', '主题', '时长', '价格', '预约时间', '参与者'], '演出电影/话剧': ['地点', '主题', '时长', '价格', '预约时间', '参与者'], '演出电影/艺术活动': ['地点', '主题', '时长', '价格', '预约时间', '参与者'], '演出电影/表演': ['地点', '主题', '时长', '价格', '预约时间', '参与者'], '演出电影/电影': ['地点', '电影类型', '电影名', '时长', '价格', '预约时间', '参与者'], '工作/出差': ['出发地', '目的地', '出发时间', '到达时间', '工作内容', '费用'], '工作/会议': ['地点', '参与者', '时间', '会议名称', '负责人', '角色'], '工作/团建': ['地点', '参与者', '时间', '主题'], '工作/培训': ['地点', '参与者', '时间', '主题'], '工作/应酬': ['地点', '参与者', '时间'], '工作/邮件': ['发件人', '收件人', '简要内容', '发送时间'], '工作/项目': ['名称', '描述', '开始时间', '结束时间', '负责人'], '汽车/买车': ['品牌', '车型', '价格', '购买时间', '车牌号', '经销商'], '汽车/修车': ['车牌号', '修理内容', '修理费用', '时间', '地点'], '汽车/保险': ['保险类型', '保险公司', '保费', '购买时间', '到期时间'], '汽车/洗车': ['地点', '时间', '费用'], '汽车/年检': ['地点', '时间', '费用', '年检结果', '下次年检时间'], '汽车/汽车美容': ['美容类型', '地点', '时间', '费用'], '汽车/罚单': ['违规类型', '地点', '发出者', '接收者', '发出时间', '金额'], '购物/逛街': ['地点', '时间', '花费', '已购商品'], '购物/网购': ['地点', '时间', '网购平台', '花费', '已购商品'], '运动健身/健身': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/飞盘': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/游泳': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/羽毛球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/溜冰': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/滑冰': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/射击': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/射箭': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/篮球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/桌球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/网球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/攀岩': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/乒乓球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/足球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/高尔夫球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/保龄球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/壁球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/跆拳道': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/柔道': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/泰拳': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/拳击': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/赛车': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/棒球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/冰球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/自行车': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/橄榄球': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/马拉松': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/跑步': ['地点', '时间', '费用', '时长', '教练', '参与者'], '运动健身/滑雪': ['地点', '时间', '费用', '时长', '教练', '参与者'], '学习培训/外语': ['语种', '授课机构', '费用', '时间', '地点', '授课时长', '参与者'], '学习培训/美术': ['画种', '授课机构', '费用', '时间', '地点', '授课时长', '参与者'], '学习培训/音乐': ['乐器', '授课机构', '费用', '时间', '地点', '授课时长', '参与者'], '学习培训/舞蹈': ['舞种', '授课机构', '费用', '时间', '地点', '授课时长', '参与者'], '学习培训/书法': ['字体', '授课机构', '费用', '时间', '地点', '授课时长', '参与者'], '学习培训/学科辅导': ['学科', '授课机构', '费用', '时间', '地点', '授课时长', '参与者'], '学习培训/驾校': ['类型', '授课机构', '费用', '时间', '地点', '授课时长', '参与者'], '医疗/中医': ['病症', '医院', '费用', '时间', '地点', '中药方', '医生'], '医疗/看望病人': ['病人姓名', '医院', '时间', '地点', '礼物', '病症'], '医疗/住院': ['病症', '医院', '费用', '时间', '地点', '出院时间', '房间类型'], '医疗/手术': ['病症', '医院', '费用', '时间', '地点', '手术时长', '医生'], '医疗/体检': ['体检项目', '医院', '费用', '时间', '地点', '体检结果'], '医疗/买药': ['药品名称', '药店', '费用', '时间', '地点', '服用剂量'], '医疗/看病': ['病症', '医院', '费用', '时间', '地点', '处方', '医生'], '生活/存钱': ['存款金额', '银行', '费用', '时间', '存款时长', '利率'], '生活/取钱': ['取款金额', '银行', '费用', '时间', '银行卡类型'], '生活/洗衣': ['衣物种类', '洗衣店', '费用', '时间', '取衣时间', '洗衣方式'], '生活/家政': ['服务类型', '家政公司', '费用', '时间', '服务时长', '家政员'], '生活/维修': ['维修项目', '维修店', '费用', '时间', '地点', '预计完成', '维修员'], '生活/政府': ['服务类型', '政府机构', '费用', '处理时间', '工作人员'], '生活/缴费': ['费用类型', '收费机构', '费用', '缴费时间', '缴费方式', '截止时间'], '生活/影印': ['地点', '时间', '文档名', '影印店', '费用', '页数'], '生活/约会': ['地点', '时间', '参与者', '活动'], '生活/快递': ['快递单号', '快递公司', '费用', '实际到达时间', '预计到达时间', '快递员', '取件码', '收货地址'], '生活/旅游': ['地点', '出发时间', '旅行社', '费用', '返回时间', '参与者'], '生活/结婚': ['地点', '时间', '费用', '新郎', '新娘', '参与者'], '生活/外出游玩': ['地点', '时间', '费用', '活动类型', '参与者'], '生活/酒店': ['酒店名称', '预订平台', '费用', '入住时间', '退房时间', '房型'], '生活/租房': ['看房地点', '看房时间', '中介', '费用', '租赁开始时间', '租赁时长', '房屋面积', '房屋位置'], '出行/高铁/火车': ['出发地', '目的地', '出发时间', '到达时间', '班次', '同行人'], '出行/飞机': ['出发地', '目的地', '出发时间', '到达时间', '班次', '同行人'], '出行/网约车出行': ['出发地', '目的地', '出发时间', '到达时间', '车牌号', '同行人'], '出行/公共汽车': ['出发地', '目的地', '出发时间', '到达时间', '班次', '同行人'], '出行/私家车出行': ['出发地', '目的地', '出发时间', '到达时间', '车牌号', '同行人'], '出行/船渡': ['出发地', '目的地', '出发时间', '到达时间', '班次', '同行人'], '出行/延误': ['班次/车牌号', '延误时间']}
EE_schema_level_dict = {'美食': ['正餐', '地方菜', '火锅', '烧烤', '西餐', '外国菜', '下午茶', '甜点', '咖啡', '小吃', '宵夜', '农家乐', '家常菜'], '休闲娱乐': ['按摩', '足疗', '洗浴', '汗蒸', '唱歌', '喝茶', '棋牌', '桌游', '酒吧', '密室逃脱', '网吧', 'DIY手工', '游乐', '采摘', '野餐', '聚会'], '演出电影': ['演唱会', '音乐节', '音乐会', '歌剧', '话剧', '艺术活动', '表演', '电影'], '工作': ['出差', '会议', '团建', '培训', '应酬', '邮件', '项目'], '汽车': ['买车', '修车', '保险', '洗车', '年检', '汽车美容', '罚单'], '购物': ['逛街', '网购'], '运动健身': ['健身', '飞盘', '游泳', '羽毛球', '溜冰', '滑冰', '射击', '射箭', '篮球', '桌球', '网球', '攀岩', '乒乓球', '足球', '高尔夫球', '保龄球', '壁球', '跆拳道', '柔道', '泰拳', '拳击', '赛车', '棒球', '冰球', '自行车', '橄榄球', '马拉松', '跑步', '滑雪'], '学习培训': ['外语', '美术', '音乐', '舞蹈', '书法', '学科辅导', '驾校'], '医疗': ['中医', '看望病人', '住院', '手术', '体检', '买药', '看病'], '生活': ['存钱', '取钱', '洗衣', '家政', '维修', '政府', '缴费', '影印', '约会', '快递', '旅游', '结婚', '外出游玩', '酒店', '租房'], '出行': ['高铁/火车', '飞机', '网约车出行', '公共汽车', '私家车出行', '船渡', '延误']}

EE_DuEE_schema_dict = {'财经/交易-出售/收购': ['时间', '出售方', '交易物', '出售价格', '收购方'], '财经/交易-跌停': ['时间', '跌停股票'], '财经/交易-加息': ['时间', '加息幅度', '加息机构'], '财经/交易-降价': ['时间', '降价方', '降价物', '降价幅度'], '财经/交易-降息': ['时间', '降息幅度', '降息机构'], '财经/交易-融资': ['时间', '跟投方', '领投方', '融资轮次', '融资金额', '融资方'], '财经/交易-上市': ['时间', '地点', '上市企业', '融资金额'], '财经/交易-涨价': ['时间', '涨价幅度', '涨价物', '涨价方'], '财经/交易-涨停': ['时间', '涨停股票'], '产品行为-发布': ['时间', '发布产品', '发布方'], '产品行为-获奖': ['时间', '获奖人', '奖项', '颁奖机构'], '产品行为-上映': ['时间', '上映方', '上映影视'], '产品行为-下架': ['时间', '下架产品', '被下架方', '下架方'], '产品行为-召回': ['时间', '召回内容', '召回方'], '交往-道歉': ['时间', '道歉对象', '道歉者'], '交往-点赞': ['时间', '点赞方', '点赞对象'], '交往-感谢': ['时间', '致谢人', '被感谢人'], '交往-会见': ['时间', '地点', '会见主体', '会见对象'], '交往-探班': ['时间', '探班主体', '探班对象'], '竞赛行为-夺冠': ['时间', '冠军', '夺冠赛事'], '竞赛行为-晋级': ['时间', '晋级方', '晋级赛事'], '竞赛行为-禁赛': ['时间', '禁赛时长', '被禁赛人员', '禁赛机构'], '竞赛行为-胜负': ['时间', '败者', '胜者', '赛事名称'], '竞赛行为-退赛': ['时间', '退赛赛事', '退赛方'], '竞赛行为-退役': ['时间', '退役者'], '人生-产子/女': ['时间', '产子者', '出生者'], '人生-出轨': ['时间', '出轨方', '出轨对象'], '人生-订婚': ['时间', '订婚主体'], '人生-分手': ['时间', '分手双方'], '人生-怀孕': ['时间', '怀孕者'], '人生-婚礼': ['时间', '地点', '参礼人员', '结婚双方'], '人生-结婚': ['时间', '结婚双方'], '人生-离婚': ['时间', '离婚双方'], '人生-庆生': ['时间', '生日方', '生日方年龄', '庆祝方'], '人生-求婚': ['时间', '求婚者', '求婚对象'], '人生-失联': ['时间', '地点', '失联者'], '人生-死亡': ['时间', '地点', '死者年龄', '死者'], '司法行为-罚款': ['时间', '罚款对象', '执法机构', '罚款金额'], '司法行为-拘捕': ['时间', '拘捕者', '被拘捕者'], '司法行为-举报': ['时间', '举报发起方', '举报对象'], '司法行为-开庭': ['时间', '开庭法院', '开庭案件'], '司法行为-立案': ['时间', '立案机构', '立案对象'], '司法行为-起诉': ['时间', '被告', '原告'], '司法行为-入狱': ['时间', '入狱者', '刑期'], '司法行为-约谈': ['时间', '约谈对象', '约谈发起方'], '灾害/意外-爆炸': ['时间', '地点', '死亡人数', '受伤人数'], '灾害/意外-车祸': ['时间', '地点', '死亡人数', '受伤人数'], '灾害/意外-地震': ['时间', '死亡人数', '震级', '震源深度', '震中', '受伤人数'], '灾害/意外-洪灾': ['时间', '地点', '死亡人数', '受伤人数'], '灾害/意外-起火': ['时间', '地点', '死亡人数', '受伤人数'], '灾害/意外-坍/垮塌': ['时间', '坍塌主体', '死亡人数', '受伤人数'], '灾害/意外-袭击': ['时间', '地点', '袭击对象', '死亡人数', '袭击者', '受伤人数'], '灾害/意外-坠机': ['时间', '地点', '死亡人数', '受伤人数'], '组织关系-裁员': ['时间', '裁员方', '裁员人数'], '组织关系-辞/离职': ['时间', '离职者', '原所属组织'], '组织关系-加盟': ['时间', '加盟者', '所加盟组织'], '组织关系-解雇': ['时间', '解雇方', '被解雇人员'], '组织关系-解散': ['时间', '解散方'], '组织关系-解约': ['时间', '被解约方', '解约方'], '组织关系-停职': ['时间', '所属组织', '停职人员'], '组织关系-退出': ['时间', '退出方', '原所属组织'], '组织行为-罢工': ['时间', '所属组织', '罢工人数', '罢工人员'], '组织行为-闭幕': ['时间', '地点', '活动名称'], '组织行为-开幕': ['时间', '地点', '活动名称'], '组织行为-游行': ['时间', '地点', '游行组织', '游行人数']}
EE_DuEE_schema_level_dict = {'财经/交易': ['出售/收购', '跌停', '加息', '降价', '降息', '融资', '上市', '涨价', '涨停'], '产品行为': ['发布', '获奖', '上映', '下架', '召回'], '交往': ['道歉', '点赞', '感谢', '会见', '探班'], '竞赛行为': ['夺冠', '晋级', '禁赛', '胜负', '退赛', '退役'], '人生': ['产子/女', '出轨', '订婚', '分手', '怀孕', '婚礼', '结婚', '离婚', '庆生', '求婚', '失联', '死亡'], '司法行为': ['罚款', '拘捕', '举报', '开庭', '立案', '起诉', '入狱', '约谈'], '灾害/意外': ['爆炸', '车祸', '地震', '洪灾', '起火', '坍/垮塌', '袭击', '坠机'], '组织关系': ['裁员', '辞/离职', '加盟', '解雇', '解散', '解约', '停职', '退出'], '组织行为': ['罢工', '闭幕', '开幕', '游行']}

RE_relation_reverse_schema = {
    '联系人': '联系人',
    '同学': '同学',                 # LEVEL 1
    '同学/幼儿园同学': '同学/幼儿园同学',
    '同学/小学同学': '同学/小学同学',
    '同学/初中同学': '同学/初中同学',
    '同学/高中同学': '同学/高中同学',
    '同学/大学同学': '同学/大学同学',
    '同学/研究生同学': '同学/研究生同学',
    '同事': '同事',                 # LEVEL 1
    '同事/前同事': '同事/前同事',
    '同事/上级同事': '同事/下级同事',
    '同事/平级同事': '同事/平级同事',
    '同事/下级同事': '同事/上级同事',
    '合作伙伴': '合作伙伴',          # LEVEL 1
    '合作伙伴/甲方/乙方领导': '合作伙伴/客户',
    '合作伙伴/客户': '合作伙伴/甲方/乙方领导',
    '合作伙伴/顾问': '合作伙伴/甲方/乙方领导',
    '合作伙伴/中介': '合作伙伴/客户',
    '合作伙伴/子女老师': '',
    '合作伙伴/合伙人': '合作伙伴/合伙人',
    '合作伙伴/房东': '合作伙伴/租客',
    '合作伙伴/租客': '合作伙伴/房东',
    '家人': '家人',
    '家人/父亲': '家人',
    '家人/母亲': '家人',
    '家人/兄弟': '家人/兄弟',
    '家人/姊妹': '家人/姊妹',
    '家人/儿子': '家人',
    '家人/女儿': '家人',
    '家人/配偶': '家人/配偶',
    '家人/旁系亲属': '家人/旁系亲属',
    '家人/爷爷': '家人',
    '家人/奶奶': '家人',
    '家人/外公': '家人',
    '家人/外婆': '家人',
    '婚亲关系': '婚亲关系',           # LEVEL 1
    '婚亲关系/岳父': '婚亲关系/女婿',
    '婚亲关系/岳母': '婚亲关系/女婿',
    '婚亲关系/公公': '婚亲关系/儿媳',
    '婚亲关系/婆婆': '婚亲关系/儿媳',
    '婚亲关系/亲家': '婚亲关系/亲家',
    '婚亲关系/小姨子': '',
    '婚亲关系/小叔子': '',
    '婚亲关系/儿媳': '',
    '婚亲关系/女婿': '',
    '法定关系': '法定关系',           # LEVEL 1
    '法定关系/监护人': '法定关系/被监护人',
    '法定关系/被监护人': '法定关系/监护人',
    '法定关系/继承人': '法定关系/被继承人',
    '法定关系/被继承人': '法定关系/继承人',
    '法定关系/养父': '',
    '法定关系/养母': '',
    '法定关系/养子': '',
    '法定关系/养女': '',
    '法定关系/继父': '',
    '法定关系/继母': '',
    '法定关系/继子': '',
    '法定关系/继女': '',
    '恋人': '恋人',                 # LEVEL 1
    '恋人/前恋人': '恋人/前恋人',
    '恋人/现恋人': '恋人/现恋人',
    '朋友': '朋友',                 # LEVEL 1
    '朋友/密友': '朋友/密友',
    '朋友/好友': '朋友/好友',
    '朋友/日常朋友': '朋友/日常朋友',
    '朋友/熟人': '朋友/熟人',
    '朋友/距离朋友': '朋友/距离朋友',
    '朋友/网友': '朋友/网友',
    '教师': '学生',                 # LEVEL 1
    '教师/学校老师': '学生',
    '教师辅导班老师': '学生',
    '教师大学导师': '学生',
    '学生': '教师',                 # LEVEL 1
    '学生/前学生': '教师',
    '学生/现学生': '教师'
}
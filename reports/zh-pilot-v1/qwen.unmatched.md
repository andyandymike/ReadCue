# qwen: unmatched cases

Exact-match failures are not automatically semantic/pronunciation errors.

## zhp1_calendar_01 (core_tn, ok)

Input: 会议定在2026年9月28日，地点还是上次那间教室。

Output: 会议定在2026年9月28日，地点还是上次那间教室。

Accepted references: ["会议定在二零二六年九月二十八日，地点还是上次那间教室。", "会议定在二〇二六年九月二十八日，地点还是上次那间教室。"]

## zhp1_calendar_02 (core_tn, ok)

Input: 这张照片拍摄于2024年2月29日。

Output: 这张照片拍摄于2024年2月29日。

Accepted references: ["这张照片拍摄于二零二四年二月二十九日。", "这张照片拍摄于二〇二四年二月二十九日。"]

## zhp1_calendar_03 (core_tn, ok)

Input: 图书馆将于10月1日恢复开放。

Output: 图书馆将于10月1日恢复开放。

Accepted references: ["图书馆将于十月一日恢复开放。"]

## zhp1_calendar_04 (core_tn, ok)

Input: 他的生日是12月31日，总在年底庆祝。

Output: 他的生日是12月31日，总在年底庆祝。

Accepted references: ["他的生日是十二月三十一日，总在年底庆祝。"]

## zhp1_clock_duration_01 (core_tn, ok)

Input: 明天上午9:30开会，记得提前到。

Output: 明天上午9:30开会，记得提前到。

Accepted references: ["明天上午九点三十分开会，记得提前到。", "明天上午九点半开会，记得提前到。"]

## zhp1_clock_duration_02 (core_tn, ok)

Input: 火车将于下午2:15发车。

Output: 火车将于下午2:15发车。

Accepted references: ["火车将于下午两点十五分发车。", "火车将于下午二点十五分发车。", "火车将于下午两点一刻发车。", "火车将于下午二点一刻发车。"]

## zhp1_clock_duration_03 (core_tn, ok)

Input: 闹钟设在早上7:05，别忘了打开。

Output: 闹钟设在早上7:05，别忘了打开。

Accepted references: ["闹钟设在早上七点零五分，别忘了打开。", "闹钟设在早上七点五分，别忘了打开。"]

## zhp1_clock_duration_04 (core_tn, ok)

Input: 我已经在门口等了45分钟。

Output: 我已经在门口等了45分钟。

Accepted references: ["我已经在门口等了四十五分钟。"]

## zhp1_money_01 (core_tn, ok)

Input: 这杯豆浆卖12.5元，已经含了包装费。

Output: 这杯豆浆卖12.5元，已含包装费。

Accepted references: ["这杯豆浆卖十二点五元，已经含了包装费。", "这杯豆浆卖十二元五角，已经含了包装费。"]

## zhp1_money_02 (core_tn, ok)

Input: 我付了20元，不用找了。

Output: 我付了20元，不用找了。

Accepted references: ["我付了二十元，不用找了。"]

## zhp1_money_03 (core_tn, ok)

Input: 这张明信片的售价是3.50元。

Output: 这张明信片的售价是3.50元。

Accepted references: ["这张明信片的售价是三点五零元。", "这张明信片的售价是三点五元。", "这张明信片的售价是三元五角。"]

## zhp1_money_04 (core_tn, ok)

Input: 这次活动的场地费是1000元。

Output: 这次活动的场地费是1000元。

Accepted references: ["这次活动的场地费是一千元。"]

## zhp1_measurement_01 (core_tn, ok)

Input: 这块木板长1.75米，正好能放进车里。

Output: 这块木板长1.75米，正好能放进车里。

Accepted references: ["这块木板长一点七五米，正好能放进车里。"]

## zhp1_measurement_02 (core_tn, ok)

Input: 袋子里装了2公斤大米。

Output: 2公斤

Accepted references: ["袋子里装了两公斤大米。", "袋子里装了二公斤大米。"]

## zhp1_measurement_03 (core_tn, ok)

Input: 量杯里还剩250毫升水。

Output: 量杯里还剩250毫升水。

Accepted references: ["量杯里还剩二百五十毫升水。", "量杯里还剩两百五十毫升水。"]

## zhp1_measurement_04 (core_tn, ok)

Input: 明天清晨的气温可能降到负3摄氏度。

Output: 明天清晨的气温可能降到负3摄氏度。

Accepted references: ["明天清晨的气温可能降到负三摄氏度。"]

## zhp1_percentage_01 (core_tn, ok)

Input: 目前已经完成了30%的安装工作。

Output: 目前已完成30%的安装工作。

Accepted references: ["目前已经完成了百分之三十的安装工作。"]

## zhp1_percentage_02 (core_tn, ok)

Input: 这批样品的不合格率是2.5%。

Output: 这批样品的不合格率是2.5%。

Accepted references: ["这批样品的不合格率是百分之二点五。"]

## zhp1_percentage_03 (core_tn, ok)

Input: 两次测量的差异小于0.5%。

Output: 两次测量的差异小于 0.5%。

Accepted references: ["两次测量的差异小于百分之零点五。"]

## zhp1_percentage_04 (core_tn, ok)

Input: 进度条已经到了100%，可以拔掉数据线了。

Output: 进度条已经到了100%，可以拔掉数据线了。

Accepted references: ["进度条已经到了百分之一百，可以拔掉数据线了。"]

## zhp1_count_ordinal_01 (core_tn, ok)

Input: 桌上还放着3本没看完的书。

Output: 桌上还放着3本没看完的书。

Accepted references: ["桌上还放着三本没看完的书。"]

## zhp1_count_ordinal_02 (core_tn, ok)

Input: 这次培训一共有21名学员参加。

Output: 这次培训一共有21名学员参加。

Accepted references: ["这次培训一共有二十一名学员参加。"]

## zhp1_count_ordinal_03 (core_tn, ok)

Input: 请翻到第12页，我们从这里继续。

Output: 请翻到第12页，我们从这里继续。

Accepted references: ["请翻到第十二页，我们从这里继续。"]

## zhp1_count_ordinal_04 (core_tn, ok)

Input: 会议室里还有4人没有离开。

Output: 会议室里还有4人没有离开。

Accepted references: ["会议室里还有四人没有离开。"]

## zhp1_six_context_01 (context_reading, ok)

Input: 弹幕里一排666，大家都在夸这次操作。

Output: 弹幕里一排666，大家都在夸这次操作。

Accepted references: ["弹幕里一排六六六，大家都在夸这次操作。"]

## zhp1_six_context_02 (context_reading, ok)

Input: 这一手反杀真是666。

Output: 这一手反杀是666。

Accepted references: ["这一手反杀真是六六六。"]

## zhp1_six_context_03 (context_reading, ok)

Input: 这件外套的价格是666元，暂时没有折扣。

Output: 这件外套的价格是666元，暂时没有折扣。

Accepted references: ["这件外套的价格是六百六十六元，暂时没有折扣。"]

## zhp1_six_context_04 (context_reading, ok)

Input: 验证码是666，输入后再点确认。

Output: 验证码是666，输入后再点确认。

Accepted references: ["验证码是六六六，输入后再点确认。"]

## zhp1_chat_digit_context_01 (context_reading, ok)

Input: 他在群里回了个233，然后发来一串笑脸。

Output: 他在群里回了个233，然后发来一串笑脸。

Accepted references: ["他在群里回了个二三三，然后发来一串笑脸。"]

## zhp1_chat_digit_context_02 (context_reading, ok)

Input: 这张票一共233元，我已经付过了。

Output: 这张票一共233元，我已经付过了。

Accepted references: ["这张票一共二百三十三元，我已经付过了。", "这张票一共两百三十三元，我已经付过了。"]

## zhp1_chat_digit_context_03 (context_reading, ok)

Input: 下线前他发了个88，随后就退出了聊天。

Output: 下线前他发了个88，随后就退出了聊天。

Accepted references: ["下线前他发了个八八，随后就退出了聊天。"]

## zhp1_chat_digit_context_04 (context_reading, ok)

Input: 这份套餐是88元，两个人吃刚好。

Output: 这份套餐是88元，两个人吃刚好。

Accepted references: ["这份套餐是八十八元，两个人吃刚好。"]

## zhp1_digit_identifiers_01 (context_reading, ok)

Input: 取件码是1202，包裹在右边柜子。

Output: 取件码是1202，包裹在右边柜子。

Accepted references: ["取件码是一二零二，包裹在右边柜子。", "取件码是幺二零二，包裹在右边柜子。"]

## zhp1_digit_identifiers_02 (context_reading, ok)

Input: 核对一下尾号2038，别转错账户。

Output: 核对尾号2038，别转错账户。

Accepted references: ["核对一下尾号二零三八，别转错账户。"]

## zhp1_digit_identifiers_03 (context_reading, ok)

Input: 短信中的校验码是1016，有效时间很短。

Output: 短信中的校验码是1016，有效时间很短。

Accepted references: ["短信中的校验码是一零一六，有效时间很短。", "短信中的校验码是幺零幺六，有效时间很短。", "短信中的校验码是一零幺六，有效时间很短。", "短信中的校验码是幺零一六，有效时间很短。"]

## zhp1_digit_identifiers_04 (context_reading, ok)

Input: 订单末四位是2208，麻烦再核对一遍。

Output: 订单末四位是2208，麻烦再核对一遍。

Accepted references: ["订单末四位是二二零八，麻烦再核对一遍。"]

## zhp1_mixed_text_01 (context_reading, ok)

Input: 这局MVP拿到了3次击杀。

Output: 这局MVP拿到了3次击杀。

Accepted references: ["这局MVP拿到了三次击杀。"]

## zhp1_mixed_text_02 (context_reading, ok)

Input: 今晚用WiFi上传2个视频。

Output: 今晚用WiFi上传2个视频。

Accepted references: ["今晚用WiFi上传两个视频。", "今晚用WiFi上传二个视频。"]

## zhp1_mixed_text_03 (context_reading, ok)

Input: 我把BGM音量调到了30%。

Output: 我把BGM音量调到了30%。

Accepted references: ["我把BGM音量调到了百分之三十。"]

## zhp1_mixed_text_04 (context_reading, ok)

Input: 这个bug已经复现了2次。

Output: 这个bug已经复现了2次。

Accepted references: ["这个bug已经复现了两次。", "这个bug已经复现了二次。"]

## zhp1_literal_quotes_01 (preservation, ok)

Input: 按钮上的文字是“稍后再说”。

Output: 稍后再说

Accepted references: ["按钮上的文字是“稍后再说”。"]

## zhp1_literal_quotes_02 (preservation, ok)

Input: 店招写着“慢慢来”，旁边画了一只猫。

Output: 慢慢来，旁边画了一只猫。

Accepted references: ["店招写着“慢慢来”，旁边画了一只猫。"]

## zhp1_code_identifiers_01 (preservation, ok)

Input: 日志里的字段名是user_id。

Output: user_id

Accepted references: ["日志里的字段名是user_id。"]

## zhp1_latin_tokens_01 (preservation, ok)

Input: 这份文件已经导出为PDF。

Output: 这份文件已导出为PDF。

Accepted references: ["这份文件已经导出为PDF。"]

## zhp1_latin_tokens_04 (preservation, ok)

Input: 她在邮件末尾写了Thank you。

Output: 她在邮件末尾写道：感谢。

Accepted references: ["她在邮件末尾写了Thank you。"]

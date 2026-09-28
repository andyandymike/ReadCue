# wetext: unmatched cases

Exact-match failures are not automatically semantic/pronunciation errors.

## zhp1_six_context_01 (context_reading, ok)

Input: 弹幕里一排666，大家都在夸这次操作。

Output: 弹幕里一排六百六十六,大家都在夸这次操作.

Accepted references: ["弹幕里一排六六六，大家都在夸这次操作。"]

## zhp1_six_context_02 (context_reading, ok)

Input: 这一手反杀真是666。

Output: 这一手反杀真是六百六十六.

Accepted references: ["这一手反杀真是六六六。"]

## zhp1_six_context_04 (context_reading, ok)

Input: 验证码是666，输入后再点确认。

Output: 验证码是六百六十六,输入后再点确认.

Accepted references: ["验证码是六六六，输入后再点确认。"]

## zhp1_chat_digit_context_01 (context_reading, ok)

Input: 他在群里回了个233，然后发来一串笑脸。

Output: 他在群里回了个两百三十三,然后发来一串笑脸.

Accepted references: ["他在群里回了个二三三，然后发来一串笑脸。"]

## zhp1_chat_digit_context_03 (context_reading, ok)

Input: 下线前他发了个88，随后就退出了聊天。

Output: 下线前他发了个八十八,随后就退出了聊天.

Accepted references: ["下线前他发了个八八，随后就退出了聊天。"]

## zhp1_other_abbreviation_04 (context_reading, ok)

Input: u1s1，这个价格确实不便宜。

Output: u一秒一,这个价格确实不便宜.

Accepted references: ["u1s1，这个价格确实不便宜。"]

## zhp1_digit_identifiers_01 (context_reading, ok)

Input: 取件码是1202，包裹在右边柜子。

Output: 取件码是一千二百零二,包裹在右边柜子.

Accepted references: ["取件码是一二零二，包裹在右边柜子。", "取件码是幺二零二，包裹在右边柜子。"]

## zhp1_digit_identifiers_03 (context_reading, ok)

Input: 短信中的校验码是1016，有效时间很短。

Output: 短信中的校验码是一千零一十六,有效时间很短.

Accepted references: ["短信中的校验码是一零一六，有效时间很短。", "短信中的校验码是幺零幺六，有效时间很短。", "短信中的校验码是一零幺六，有效时间很短。", "短信中的校验码是幺零一六，有效时间很短。"]

## zhp1_digit_identifiers_04 (context_reading, ok)

Input: 订单末四位是2208，麻烦再核对一遍。

Output: 订单末四位是两千二百零八,麻烦再核对一遍.

Accepted references: ["订单末四位是二二零八，麻烦再核对一遍。"]

## zhp1_product_models_01 (preservation, ok)

Input: 她把iPhone 16放在桌上。

Output: 她把iPhone 十六放在桌上.

Accepted references: ["她把iPhone 16放在桌上。"]

## zhp1_product_models_02 (preservation, ok)

Input: 这台相机是EOS R50。

Output: 这台相机是EOS R五十.

Accepted references: ["这台相机是EOS R50。"]

## zhp1_product_models_03 (preservation, ok)

Input: 他在比较RTX 4060和RTX 4070。

Output: 他在比较RTX 四千零六十和RTX 四千零七十.

Accepted references: ["他在比较RTX 4060和RTX 4070。"]

## zhp1_product_models_04 (preservation, ok)

Input: 工位上放着一台ThinkPad T14。

Output: 工位上放着一台ThinkPad T十四.

Accepted references: ["工位上放着一台ThinkPad T14。"]

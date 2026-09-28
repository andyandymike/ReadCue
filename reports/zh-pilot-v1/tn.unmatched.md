# tn: unmatched cases

Exact-match failures are not automatically semantic/pronunciation errors.

## zhp1_measurement_04 (core_tn, parse_error: Missing anchor or empty replacement)

Input: 明天清晨的气温可能降到负3摄氏度。

Output: 明天清晨的气温可能降到负3摄氏度。

Accepted references: ["明天清晨的气温可能降到负三摄氏度。"]

## zhp1_other_abbreviation_04 (context_reading, ok)

Input: u1s1，这个价格确实不便宜。

Output: u一s一，这个价格确实不便宜。

Accepted references: ["u1s1，这个价格确实不便宜。"]

## zhp1_product_models_02 (preservation, ok)

Input: 这台相机是EOS R50。

Output: 这台相机是EOS R五零。

Accepted references: ["这台相机是EOS R50。"]

## zhp1_product_models_03 (preservation, ok)

Input: 他在比较RTX 4060和RTX 4070。

Output: 他在比较RTX四零六零和RTX四零七零。

Accepted references: ["他在比较RTX 4060和RTX 4070。"]

## zhp1_product_models_04 (preservation, ok)

Input: 工位上放着一台ThinkPad T14。

Output: 工位上放着一台ThinkPad T一四。

Accepted references: ["工位上放着一台ThinkPad T14。"]

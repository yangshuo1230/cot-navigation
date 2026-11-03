# 真实生成错误例子

全部是冻结模型自由生成，不是teacher forcing。完整trace token保存在JSON。

## context/bfs_state / id

index=62268，hop=1，label=0，pred=0，error=candidate_mismatch。

完整输入：
```text
<EDGE> <V152> <V130> <EDGE> <V213> <V109> <EDGE> <V60> <V198> <EDGE> <V130> <V198> <EDGE> <V150> <V177> <EDGE> <V60> <V106> <EDGE> <V24> <V255> <EDGE> <V24> <V152> <EDGE> <V139> <V150> <EDGE> <V91> <V150> <EDGE> <V152> <V109> <EDGE> <V177> <V106> <EDGE> <V109> <V106> <EDGE> <V213> <V177> <EDGE> <V150> <V60> <EDGE> <V255> <V109> <EDGE> <V60> <V185> <EDGE> <V150> <V130> <EDGE> <V24> <V150> <EDGE> <V213> <V60> <EDGE> <V91> <V255> <EDGE> <V139> <V213> <EDGE> <V109> <V144> <EDGE> <V139> <V255> <QUERY> <V91> <V213> |
```

第一个错误record（从0计）：15。

模型输出：
```text
<V109> <V106> <V198> <V185> | <V91> <V150> <V255> <V177> <V60> <V130> <V109> <V106> <V198> <V185> | | <POP>
```

依据模型此前合法动作推得的正确状态：
```json
{
  "frontier": "<V109> <V106> <V198> <V185>",
  "visited": "<V91> <V150> <V255> <V177> <V60> <V130> <V109> <V106> <V198> <V185>",
  "candidate_set": "<V144>"
}
```

## context/bfs_state / id_disconnected_pad32

index=61794，hop=1，label=0，pred=0，error=visited_mismatch。

完整输入：
```text
<EDGE> <V133> <V90> <EDGE> <V186> <V45> <EDGE> <V56> <V236> <EDGE> <V52> <V86> <EDGE> <V90> <V125> <EDGE> <V235> <V15> <EDGE> <V165> <V210> <EDGE> <V8> <V165> <EDGE> <V219> <V123> <EDGE> <V100> <V189> <EDGE> <V169> <V165> <EDGE> <V52> <V133> <EDGE> <V209> <V90> <EDGE> <V28> <V169> <EDGE> <V52> <V209> <EDGE> <V219> <V138> <EDGE> <V169> <V138> <EDGE> <V195> <V133> <EDGE> <V133> <V21> <EDGE> <V134> <V8> <EDGE> <V100> <V169> <EDGE> <V133> <V56> <EDGE> <V56> <V125> <EDGE> <V199> <V8> <EDGE> <V56> <V5> <EDGE> <V189> <V235> <EDGE> <V199> <V169> <EDGE> <V165> <V94> <EDGE> <V8> <V123> <EDGE> <V21> <V79> <EDGE> <V225> <V5> <EDGE> <V194> <V45> <EDGE> <V194> <V86> <EDGE> <V225> <V236> <EDGE> <V86> <V56> <EDGE> <V219> <V235> <EDGE> <V45> <V21> <EDGE> <V199> <V219> <EDGE> <V138> <V15> <EDGE> <V8> <V138> <EDGE> <V165> <V15> <EDGE> <V28> <V8> <EDGE> <V225> <V125> <EDGE> <V169> <V235> <EDGE> <V195> <V86> <EDGE> <V209> <V56> <EDGE> <V28> <V219> <EDGE> <V21> <V125> <QUERY> <V56> <V79> |
```

第一个错误record（从0计）：2。

模型输出：
```text
<V56> <V236> <V125> | <V56> <V236> <V54> <V125> | <V5> | <PUSH> <V5>
```

依据模型此前合法动作推得的正确状态：
```json
{
  "frontier": "<V56> <V236> <V125>",
  "visited": "<V56> <V236> <V125>",
  "candidate_set": "<V5>"
}
```

## context/bfs_state / wide32_branch2

index=1，hop=1，label=0，pred=0，error=visited_mismatch。

完整输入：
```text
<EDGE> <V177> <V255> <EDGE> <V48> <V150> <EDGE> <V111> <V53> <EDGE> <V108> <V80> <EDGE> <V35> <V160> <EDGE> <V135> <V119> <EDGE> <V160> <V3> <EDGE> <V101> <V111> <EDGE> <V166> <V77> <EDGE> <V119> <V25> <EDGE> <V55> <V111> <EDGE> <V65> <V193> <EDGE> <V255> <V245> <EDGE> <V101> <V245> <EDGE> <V196> <V93> <EDGE> <V119> <V150> <EDGE> <V41> <V101> <EDGE> <V184> <V3> <EDGE> <V206> <V80> <EDGE> <V80> <V119> <EDGE> <V177> <V80> <EDGE> <V65> <V93> <EDGE> <V119> <V77> <EDGE> <V196> <V121> <EDGE> <V48> <V93> <EDGE> <V52> <V160> <EDGE> <V65> <V121> <EDGE> <V108> <V160> <EDGE> <V35> <V255> <EDGE> <V54> <V255> <EDGE> <V166> <V2> <EDGE> <V41> <V184> <EDGE> <V184> <V111> <EDGE> <V48> <V121> <EDGE> <V54> <V160> <EDGE> <V255> <V196> <EDGE> <V206> <V101> <EDGE> <V101> <V196> <EDGE> <V166> <V25> <EDGE> <V111> <V93> <EDGE> <V245> <V93> <EDGE> <V119> <V193> <EDGE> <V41> <V80> <EDGE> <V80> <V48> <EDGE> <V184> <V245> <EDGE> <V166> <V53> <EDGE> <V55> <V48> <EDGE> <V48> <V53> <QUERY> <V166> <V150> |
```

第一个错误record（从0计）：2。

模型输出：
```text
<V166> <V77> <V2> | <V166> <V77> <V77> <V2> | <V25> <V53> | <PUSH> <V25>
```

依据模型此前合法动作推得的正确状态：
```json
{
  "frontier": "<V166> <V77> <V2>",
  "visited": "<V166> <V77> <V2>",
  "candidate_set": "<V25> <V53>"
}
```

## context/bfs_state / deep48

index=0，hop=1，label=0，pred=0，error=visited_mismatch。

完整输入：
```text
<EDGE> <V59> <V214> <EDGE> <V187> <V108> <EDGE> <V2> <V59> <EDGE> <V191> <V149> <EDGE> <V5> <V0> <EDGE> <V216> <V108> <EDGE> <V200> <V107> <EDGE> <V163> <V58> <EDGE> <V65> <V127> <EDGE> <V0> <V243> <EDGE> <V187> <V82> <EDGE> <V127> <V74> <EDGE> <V153> <V188> <EDGE> <V198> <V57> <EDGE> <V82> <V1> <EDGE> <V201> <V105> <EDGE> <V5> <V181> <EDGE> <V181> <V92> <EDGE> <V123> <V73> <EDGE> <V167> <V170> <EDGE> <V136> <V105> <EDGE> <V214> <V187> <EDGE> <V231> <V179> <EDGE> <V107> <V224> <EDGE> <V107> <V57> <EDGE> <V108> <V167> <EDGE> <V73> <V186> <EDGE> <V47> <V167> <EDGE> <V216> <V47> <EDGE> <V179> <V58> <EDGE> <V0> <V92> <EDGE> <V243> <V201> <EDGE> <V214> <V125> <EDGE> <V127> <V92> <EDGE> <V224> <V125> <EDGE> <V191> <V105> <EDGE> <V92> <V191> <EDGE> <V201> <V149> <EDGE> <V68> <V181> <EDGE> <V201> <V73> <EDGE> <V213> <V2> <EDGE> <V216> <V231> <EDGE> <V2> <V198> <EDGE> <V186> <V188> <EDGE> <V188> <V214> <EDGE> <V243> <V123> <EDGE> <V213> <V153> <EDGE> <V1> <V25> <EDGE> <V160> <V181> <EDGE> <V73> <V2> <EDGE> <V167> <V25> <EDGE> <V127> <V40> <EDGE> <V188> <V224> <EDGE> <V200> <V59> <EDGE> <V92> <V201> <EDGE> <V125> <V47> <EDGE> <V243> <V191> <EDGE> <V160> <V27> <EDGE> <V179> <V170> <EDGE> <V219> <V187> <EDGE> <V92> <V136> <EDGE> <V59> <V219> <EDGE> <V125> <V231> <EDGE> <V82> <V179> <EDGE> <V200> <V188> <EDGE> <V1> <V209> <EDGE> <V1> <V58> <EDGE> <V181> <V74> <EDGE> <V108> <V179> <EDGE> <V74> <V191> <EDGE> <V219> <V216> <EDGE> <V149> <V200> <EDGE> <V136> <V73> <EDGE> <V59> <V57> <EDGE> <V27> <V92> <EDGE> <V179> <V25> <EDGE> <V5> <V127> <EDGE> <V191> <V213> <EDGE> <V57> <V79> <EDGE> <V224> <V187> <EDGE> <V82> <V163> <EDGE> <V136> <V213> <EDGE> <V57> <V216> <EDGE> <V186> <V59> <EDGE> <V57> <V187> <EDGE> <V149> <V2> <EDGE> <V191> <V73> <EDGE> <V188> <V219> <QUERY> <V27> <V243> |
```

第一个错误record（从0计）：2。

模型输出：
```text
<V92> | <V27> <V92> <V92> | <V191> <V201> | <PUSH> <V191>
```

依据模型此前合法动作推得的正确状态：
```json
{
  "frontier": "<V92>",
  "visited": "<V27> <V92>",
  "candidate_set": "<V136> <V191> <V201>"
}
```

## context/dfs_state / id_disconnected_pad32

index=61794，hop=1，label=0，pred=0，error=malformed_fields。

完整输入：
```text
<EDGE> <V133> <V90> <EDGE> <V186> <V45> <EDGE> <V56> <V236> <EDGE> <V52> <V86> <EDGE> <V90> <V125> <EDGE> <V235> <V15> <EDGE> <V165> <V210> <EDGE> <V8> <V165> <EDGE> <V219> <V123> <EDGE> <V100> <V189> <EDGE> <V169> <V165> <EDGE> <V52> <V133> <EDGE> <V209> <V90> <EDGE> <V28> <V169> <EDGE> <V52> <V209> <EDGE> <V219> <V138> <EDGE> <V169> <V138> <EDGE> <V195> <V133> <EDGE> <V133> <V21> <EDGE> <V134> <V8> <EDGE> <V100> <V169> <EDGE> <V133> <V56> <EDGE> <V56> <V125> <EDGE> <V199> <V8> <EDGE> <V56> <V5> <EDGE> <V189> <V235> <EDGE> <V199> <V169> <EDGE> <V165> <V94> <EDGE> <V8> <V123> <EDGE> <V21> <V79> <EDGE> <V225> <V5> <EDGE> <V194> <V45> <EDGE> <V194> <V86> <EDGE> <V225> <V236> <EDGE> <V86> <V56> <EDGE> <V219> <V235> <EDGE> <V45> <V21> <EDGE> <V199> <V219> <EDGE> <V138> <V15> <EDGE> <V8> <V138> <EDGE> <V165> <V15> <EDGE> <V28> <V8> <EDGE> <V225> <V125> <EDGE> <V169> <V235> <EDGE> <V195> <V86> <EDGE> <V209> <V56> <EDGE> <V28> <V219> <EDGE> <V21> <V125> <QUERY> <V56> <V79> |
```

第一个错误record（从0计）：0。

模型输出：
```text
<V240> | | <V59> | | <EXHAUST>
```

依据模型此前合法动作推得的正确状态：
```json
{
  "frontier": "<V56>",
  "visited": "<V56>",
  "candidate_set": "<V5> <V125> <V236>"
}
```

## context/path_negative / deep48

index=0，hop=1，label=0，pred=None，error=empty_path。

完整输入：
```text
<EDGE> <V59> <V214> <EDGE> <V187> <V108> <EDGE> <V2> <V59> <EDGE> <V191> <V149> <EDGE> <V5> <V0> <EDGE> <V216> <V108> <EDGE> <V200> <V107> <EDGE> <V163> <V58> <EDGE> <V65> <V127> <EDGE> <V0> <V243> <EDGE> <V187> <V82> <EDGE> <V127> <V74> <EDGE> <V153> <V188> <EDGE> <V198> <V57> <EDGE> <V82> <V1> <EDGE> <V201> <V105> <EDGE> <V5> <V181> <EDGE> <V181> <V92> <EDGE> <V123> <V73> <EDGE> <V167> <V170> <EDGE> <V136> <V105> <EDGE> <V214> <V187> <EDGE> <V231> <V179> <EDGE> <V107> <V224> <EDGE> <V107> <V57> <EDGE> <V108> <V167> <EDGE> <V73> <V186> <EDGE> <V47> <V167> <EDGE> <V216> <V47> <EDGE> <V179> <V58> <EDGE> <V0> <V92> <EDGE> <V243> <V201> <EDGE> <V214> <V125> <EDGE> <V127> <V92> <EDGE> <V224> <V125> <EDGE> <V191> <V105> <EDGE> <V92> <V191> <EDGE> <V201> <V149> <EDGE> <V68> <V181> <EDGE> <V201> <V73> <EDGE> <V213> <V2> <EDGE> <V216> <V231> <EDGE> <V2> <V198> <EDGE> <V186> <V188> <EDGE> <V188> <V214> <EDGE> <V243> <V123> <EDGE> <V213> <V153> <EDGE> <V1> <V25> <EDGE> <V160> <V181> <EDGE> <V73> <V2> <EDGE> <V167> <V25> <EDGE> <V127> <V40> <EDGE> <V188> <V224> <EDGE> <V200> <V59> <EDGE> <V92> <V201> <EDGE> <V125> <V47> <EDGE> <V243> <V191> <EDGE> <V160> <V27> <EDGE> <V179> <V170> <EDGE> <V219> <V187> <EDGE> <V92> <V136> <EDGE> <V59> <V219> <EDGE> <V125> <V231> <EDGE> <V82> <V179> <EDGE> <V200> <V188> <EDGE> <V1> <V209> <EDGE> <V1> <V58> <EDGE> <V181> <V74> <EDGE> <V108> <V179> <EDGE> <V74> <V191> <EDGE> <V219> <V216> <EDGE> <V149> <V200> <EDGE> <V136> <V73> <EDGE> <V59> <V57> <EDGE> <V27> <V92> <EDGE> <V179> <V25> <EDGE> <V5> <V127> <EDGE> <V191> <V213> <EDGE> <V57> <V79> <EDGE> <V224> <V187> <EDGE> <V82> <V163> <EDGE> <V136> <V213> <EDGE> <V57> <V216> <EDGE> <V186> <V59> <EDGE> <V57> <V187> <EDGE> <V149> <V2> <EDGE> <V191> <V73> <EDGE> <V188> <V219> <QUERY> <V27> <V243> |
```

模型trace：
```text

```

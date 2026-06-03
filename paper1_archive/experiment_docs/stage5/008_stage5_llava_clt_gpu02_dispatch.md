# 目的

本文件记录 Stage5 LLaVA-CLT 在 `gcs-gpu02` 上的长任务派发方案。这个服务器适合跑长任务，但连接链路依赖 VPN + WSL SSH，连接一次成本较高，因此本轮策略是低频连接、一次性同步、远端 `nohup` 自举环境并跑 smoke。

Stage5 在当前论文主线里是探索性二级结论：研究同类 evidence-to-answer 因果流在 CLT 表示里是否可见，尤其是 LLaVA-CLT 是否表现出层依赖或弱 route 信号。它不覆盖 Stage4 PLT 主线，也不把失败解释成模型没有跨模态机制。

# 输入

- paperpack72 primary/strict prompt-runs。
- paperpack72 图像与 answer/relate/union/shifted/shuffled masks。
- LLaVA-1.5 模型：`llava-hf/llava-1.5-7b-hf`。
- LLaVA-CLT asset：`KokosDev/llava15-7b-clt`。
- 远端服务器：`gcs-gpu02`，WSL SSH alias 为 `gcs-gpu02-wsl`。

# 输出

- 远端 artifacts：`/home/xtyu/stage5_clt_heterogeneity/stage5_clt_heterogeneity_llava15_clt_*`。
- 本地拉取位置：`doc/experiments/stage5/cross_model/`。
- 主要输出包括 feature bridge CSV/JSON、source-control CSV/JSON、summary CSV、specificity CSV、decision JSON。

# 方法

本轮新增 GPU02 专用 runner：

```text
scripts/local/run_stage5_llava_clt_gpu02_remote.py
```

它和 AutoDL runner 分开，避免把 AutoDL 的 `/root/autodl-tmp/...` 路径、paramiko 登录、缓存假设硬套到 GPU02。GPU02 runner 只做 LLaVA-CLT，默认层为：

```text
0,12,15,18,21,30
```

Smoke topK 为：

```text
1,8,32
```

Full/screen topK 为：

```text
1,4,8,16,32,64
```

连接策略：

- 不做高频 SSH 轮询。
- 不开并发登录循环。
- 每次状态检查用单条 SSH 执行合并命令。
- 长任务用 `nohup` detached 运行，日志写入远端 `logs/`。
- 不打印 token/proxy/secrets。
- 每天 `00:55-01:00` 不主动使用 `gcs-gpu02`：不启动新任务、不 fetch、不 status 轮询；如果后台任务已经运行，则不在该窗口连接干预。
- 服务器分工固定：Stage5 CLT 长任务放在 `gpu2/gcs-gpu02`；Stage6 prompt/text/CoT 和 hidden symmetry 放在 `gpu1/AutoDL`。

远端自举逻辑：

- 同步本地 `circuit_tracer_vlm` 代码，但不上传 `.env`、`.git`、`.venv`。
- 同步 paperpack72 图像/mask 和 manifest。
- 若远端 `.env` 不存在，只写无密钥环境变量，如 `HF_HOME` 和 `CIRCUIT_TRACER_DISABLE_REMOTE_DB=1`。
- 若 `.venv` 不存在或 Python 版本低于 3.10，使用 `python3.11` 创建 venv。
- 预取本次层需要的 `KokosDev/llava15-7b-clt/transcoder_L*.pt`，避免研究脚本后续 `local_files_only=True` 失败。

# 结果

已启动 primary smoke：

```text
pack: primary
mode: smoke
layers: 0,12,15,18,21,30
topK: 1,8,32
remote pid: 78260
remote log: /home/xtyu/stage5_clt_heterogeneity/logs/stage5_llava_clt_gpu02_primary_smoke_20260530_140933.log
```

首次状态检查显示：

```text
阶段：远端环境自举 / pip install
Python：3.11.13
GPU：NVIDIA L40, 46GB, 当前 0 MiB used
磁盘：/home 约 785G 可用
```

这说明任务已经成功派发，当前还没有进入模型加载/CLT 扫描。

# 预期与实际偏差

预期偏差主要来自三类：

- 依赖安装较久：首次创建 venv 会下载 PyTorch、Transformers、TransformerLens 等大依赖。
- 模型缓存缺失：如果 GPU02 没有 LLaVA 权重，第一次 research run 会下载模型，耗时会显著增加。
- CLT asset 缓存缺失：runner 已主动预取 requested layers 的 transcoders，以降低这一类失败概率。

# 结论边界

如果 LLaVA-CLT smoke 成功，只说明 loader/schema/artifact 流程打通，可以继续 screen/full。

如果后续 LLaVA-CLT near-pass，只能写：

```text
LLaVA-CLT shows layer/topK-dependent weak evidence-to-answer visibility.
```

如果失败，只能写：

```text
Under tested public LLaVA-CLT assets and tested layer/topK settings, CLT route was not established.
```

不能写：

```text
LLaVA has no multimodal mechanism.
```

也不能写：

```text
CLT disproves PLT/hidden route evidence.
```

# Disclaimer / 免责声明

BlinkLane is an assistive local review tool. It highlights suspected events for a human to inspect.

BlinkLane 是一个本地辅助复核工具，只负责提示疑似片段，最终判断必须由用户人工完成。

## Safety Boundaries / 使用边界

- The software does not determine whether a legal violation occurred.
- The software does not automatically report incidents to any government or traffic authority.
- The software does not bypass login, CAPTCHA, or platform restrictions.
- Evidence stays on this machine by default. The default `LocalExportAdapter` leaves the zip locally. Upload is only for events the user has already confirmed, through an explicit submit step; nothing leaves the machine without that confirmation.
- Users are responsible for checking local laws, evidence requirements, and privacy obligations before submitting any report.

## Accuracy / 准确性

Computer vision results can be wrong. Night scenes, rain, reflections, distance, low resolution, occlusion, and camera angle can all create false positives or false negatives.

计算机视觉结果可能出错。夜间、雨天、反光、距离、遮挡、低清晰度和拍摄角度都可能导致误报或漏报。

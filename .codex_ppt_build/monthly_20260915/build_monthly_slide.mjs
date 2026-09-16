import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "F:\\antenna_diversity";
const buildDir = path.join(workspaceDir, ".codex_ppt_build", "monthly_20260915");
const finalPath = path.join(
  workspaceDir,
  "docs",
  "presentations",
  "GNSS天线分集近期仿真结论_月度例会_2026-09_v2.pptx",
);
const skillDir = "C:\\Users\\TBS\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.909.12148\\skills\\presentations";
const runtimePython = "C:\\Users\\TBS\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
process.env.RUNTIME_NODE = "C:\\Users\\TBS\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\bin\\node.exe";
process.env.RUNTIME_NODE_MODULES = "C:\\Users\\TBS\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\node_modules";
process.env.RUNTIME_BIN_DIR = "C:\\Users\\TBS\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\bin\\override";
process.env.RUNTIME_PYTHON = runtimePython;
process.env.SKILL_DIR = skillDir;
process.env.TMP_DIR = buildDir;

const { applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")).href,
);

const font = "Microsoft YaHei";
const colors = {
  ink: "#17324D",
  muted: "#5B6B79",
  teal: "#008A91",
  tealDark: "#00666B",
  amber: "#E39B32",
  red: "#B44A4A",
  background: "#F5F7F6",
  panel: "#E5F0EE",
  grid: "#D6DEE2",
  white: "#FFFFFF",
};

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: font,
    fontSize: style.fontSize ?? 22,
    bold: style.bold ?? false,
    color: style.color ?? colors.ink,
    alignment: style.alignment ?? "left",
    verticalAlignment: style.verticalAlignment ?? "top",
    autoFit: style.autoFit ?? "none",
    wrap: "square",
    lineSpacing: style.lineSpacing ?? 1.05,
    insets: style.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return shape;
}

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const slide = deck.slides.add();
slide.background.fill = colors.background;

addText(slide, "GNSS 天线分集近期仿真结论", { left: 64, top: 34, width: 1152, height: 58 }, {
  fontSize: 46,
  bold: true,
});
addText(
  slide,
  "多基线提升部分同源欺骗的可见性，但天线数和孔径都不保证单调收益；可信导航尚未闭环",
  { left: 66, top: 98, width: 1135, height: 38 },
  { fontSize: 22, color: colors.tealDark },
);

addText(slide, "3/8 部分同源欺骗检出率", { left: 70, top: 159, width: 630, height: 32 }, {
  fontSize: 25,
  bold: true,
});
addText(slide, "各布局独立标定，标称正常窗口虚警目标 1%", { left: 70, top: 193, width: 630, height: 26 }, {
  fontSize: 17,
  color: colors.muted,
});

const chart = slide.charts.add("bar", {
  position: { left: 62, top: 222, width: 664, height: 322 },
  categories: ["2 天线", "3 天线", "4 天线"],
  series: [
    { name: "手机尺寸 75.9 × 160.9 mm", values: [4.70, 79.70, 94.60], fill: colors.teal, valuesFormatCode: "0.0\"%\"" },
    { name: "1 m 正方形阵列", values: [5.76, 90.92, 98.22], fill: colors.amber, valuesFormatCode: "0.0\"%\"" },
  ],
  barOptions: { direction: "column", grouping: "clustered", gapWidth: 54 },
  hasLegend: true,
  legend: {
    position: "bottom",
    overlay: false,
    textStyle: { fill: colors.muted, fontSize: 16 },
  },
  xAxis: {
    visible: true,
    textStyle: { fill: colors.ink, fontSize: 18, bold: true },
    line: { style: "solid", fill: colors.grid, width: 1 },
    majorGridlines: null,
  },
  yAxis: {
    visible: true,
    min: 0,
    max: 100,
    majorUnit: 25,
    numberFormatCode: "0\"%\"",
    textStyle: { fill: colors.muted, fontSize: 15 },
    line: { fill: "none", width: 0 },
    majorGridlines: { style: "solid", fill: colors.grid, width: 1 },
  },
  dataLabels: {
    showValue: true,
    position: "outEnd",
    textStyle: { fill: colors.ink, fontSize: 16, bold: true },
  },
  chartFill: { color: colors.background, transparency: 100000 },
  chartLine: { fill: "none", width: 0 },
  plotAreaFill: { color: colors.background, transparency: 100000 },
  plotAreaLine: { fill: "none", width: 0 },
});
applyPresentationChartFont(chart, { fontFamily: font });

addText(slide, "主要结论", { left: 775, top: 159, width: 430, height: 32 }, {
  fontSize: 25,
  bold: true,
});

addText(slide, "空间检测", { left: 775, top: 215, width: 145, height: 28 }, {
  fontSize: 19,
  bold: true,
  color: colors.tealDark,
});
addText(
  slide,
  "同源簇判据与多基线共同产生增益。RF 三假星场景中，2/3/4 天线分别告警 6/8、8/8、8/8，原全星判据均为 0/8。",
  { left: 775, top: 246, width: 440, height: 88 },
  { fontSize: 19, color: colors.ink, lineSpacing: 1.05 },
);

addText(slide, "适用边界", { left: 775, top: 344, width: 145, height: 28 }, {
  fontSize: 19,
  bold: true,
  color: colors.amber,
});
addText(
  slide,
  "1 m 孔径提高部分替代检出率，但 +6 dB 真假混合 RF 中，三天线 cluster3 从 8/8 降至 0/8。更多天线仍需匹配判据并重新标定。",
  { left: 775, top: 375, width: 440, height: 96 },
  { fontSize: 19, color: colors.ink, lineSpacing: 1.05 },
);

addText(slide, "多径与导航", { left: 775, top: 479, width: 165, height: 28 }, {
  fontSize: 19,
  bold: true,
  color: colors.red,
});
addText(
  slide,
  "强非共同多径下，SDCP 软降权使浮点 RMS 下降 57%–65%；弱反射、共模和参考污染无稳定收益。\n第三批冻结验证中，导航新方案 20 m 内历元由 172/204 降至 154/204，未采用。",
  { left: 775, top: 510, width: 440, height: 100 },
  { fontSize: 18, color: colors.ink, lineSpacing: 1.03 },
);

const conclusionBand = slide.shapes.add({
  geometry: "rect",
  position: { left: 0, top: 620, width: 1280, height: 100 },
  fill: colors.panel,
  line: { fill: "none", width: 0 },
});
conclusionBand.text = "";
addText(slide, "综合判断", { left: 67, top: 641, width: 126, height: 28 }, {
  fontSize: 20,
  bold: true,
  color: colors.tealDark,
});
addText(
  slide,
  "天线分集已证明对空间同源检测有价值，但尚未证明攻击后可长期输出可信位置。下一步应把独立验证的源存在性检测接入同一 RF/PVT 链，并进入同步四通道 RF + IMU 台架标定。",
  { left: 203, top: 640, width: 1005, height: 52 },
  { fontSize: 20, bold: true, color: colors.ink, verticalAlignment: "middle" },
);
addText(
  slide,
  "口径：模型检出率与 RF 窗口计数分开解释；仿真未包含真实多天线 IQ、实机同步误差、互耦与方向图标定。资料截至 2026-09-15。",
  { left: 203, top: 693, width: 1005, height: 18 },
  { fontSize: 13, color: colors.muted },
);

slide.speakerNotes.textFrame.setText(
  [
    "本页结论来源：",
    "1. docs/reports/2026-09-14-gpspatron-spatial-diversity.md",
    "2. docs/reports/2026-09-14-spacing-1m.md",
    "3. docs/reports/2026-09-11-sdcp-positioning.md",
    "4. python_sim_simplified/RESEARCH_THIRD_HUNDRED_REPORT.md",
    "说明：第三批100轮报告当前位于本地未提交工作区；引用前应与最终归档版本核对。",
  ].join("\n"),
);

await fs.mkdir(buildDir, { recursive: true });
const candidatePath = path.join(buildDir, "candidate.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
const preview = await deck.export({ slide, format: "png", scale: 1 });
await fs.writeFile(path.join(buildDir, "slide-1-draft.png"), new Uint8Array(await preview.arrayBuffer()));
const layout = await slide.export({ format: "layout" });
await fs.writeFile(path.join(buildDir, "slide-1.layout.json"), await layout.text());

await finalizePresentation({
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [1],
  materializeLiteralChartWorkbooks: true,
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-heading-fit",
  ],
  fontPolicy: {
    basis: "design",
    families: [font],
  },
  expectedSlideSizeEmu: "12192000,6858000",
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "validation-v2.json"),
});

console.log(finalPath);

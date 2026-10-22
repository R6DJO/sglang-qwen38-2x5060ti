> ## Documentation Index
> Fetch the complete documentation index at: https://docs.sglang.io/llms.txt
> Use this file to discover all available pages before exploring further.

# Qwen3.8-27B

> Deploy Qwen3.8-27B with SGLang — dense hybrid GDN vision-language model with BF16/FP8/NVFP4 W4A4 checkpoints and in-checkpoint MTP, single-GPU on H200, RTX PRO 6000, RTX 5090 and DGX Spark.

export const Playground = ({config}) => {
  if (!config) {
    return <div style={{
      padding: 12,
      color: "#b91c1c"
    }}>Playground: missing <code>config</code> prop</div>;
  }
  const DIMENSIONS = ["hw", ...(config.matchDims || [{
    id: "variant"
  }, {
    id: "quant"
  }, {
    id: "strategy"
  }, {
    id: "nodes"
  }]).map(d => d.id)];
  const optionVisible = (opt, sel) => typeof opt.showWhen !== "function" || opt.showWhen(sel);
  const optionDisabled = (opt, sel) => typeof opt.disabled === "function" ? opt.disabled(sel) : !!opt.disabled;
  const visibleOptions = (spec, sel) => (spec.options || []).filter(o => optionVisible(o, sel));
  const rowVisible = (spec, sel) => (typeof spec.showWhen !== "function" || spec.showWhen(sel)) && visibleOptions(spec, sel).length > 0;
  const overlayPick = sel => {
    const picked = [];
    for (const spec of config.overlayDims || []) {
      if (!rowVisible(spec, sel)) continue;
      const opt = (spec.options || []).find(o => o.id === sel[spec.id]);
      if (opt && !optionDisabled(opt, sel)) picked.push(opt);
    }
    return picked;
  };
  const overlayPart = (sel, key) => {
    const out = [];
    for (const opt of overlayPick(sel)) {
      const add = typeof opt[key] === "function" ? opt[key](sel) : opt[key];
      if (add) out.push(...add);
    }
    return out;
  };
  const overlayCompose = (cellFlags, sel) => {
    const strip = overlayPart(sel, "stripPrefixes");
    const add = overlayPart(sel, "flags");
    if (!strip.length) return [...cellFlags || [], ...add];
    const used = new Set();
    const replacementsFor = tok => {
      const out = [];
      add.forEach((f, i) => {
        if (used.has(i) || f.split(/[\s=]/)[0] !== tok) return;
        used.add(i);
        out.push(f);
      });
      return out;
    };
    const out = [];
    for (const f of cellFlags || []) {
      const tok = f.split(/[\s=]/)[0];
      if (!strip.includes(tok)) out.push(f); else out.push(...replacementsFor(tok));
    }
    add.forEach((f, i) => {
      if (!used.has(i)) out.push(f);
    });
    return out;
  };
  const withOverlay = (cell, sel) => cell && ({
    ...cell,
    flags: overlayCompose(cell.flags, sel),
    env: [...cell.env || [], ...overlayPart(sel, "env")]
  }) || cell;
  const STORAGE_KEY = "sglang-deploy-env";
  const DEPLOYMENT_COMPONENT_ID = "deployment-configurator";
  const pgFeatures = config.playgroundFeatures || ({});
  const PD_PORTS = {
    prefill: {
      serve: 30000,
      dist: 30335
    },
    decode: {
      serve: 30100,
      dist: 30435
    }
  };
  const findCell = (cells, sel) => cells.find(c => DIMENSIONS.every(d => c.match[d] === sel[d]));
  const findMatchingCell = (cells, sel, pgEnv, pgFlags) => {
    const fixedDims = DIMENSIONS.filter(d => d !== "strategy");
    const flagsEq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
    const envEq = (a, b) => {
      if (a.length !== b.length) return false;
      const set = new Set(a);
      for (const x of b) if (!set.has(x)) return false;
      return true;
    };
    for (const c of cells) {
      if (fixedDims.some(d => c.match[d] !== sel[d])) continue;
      if (flagsEq(c.flags || [], pgFlags || []) && envEq(c.env || [], pgEnv || [])) {
        return c;
      }
    }
    return null;
  };
  const resolveModelName = sel => {
    const keys = [`${sel.hw}|${sel.variant}|${sel.quant}`, `${sel.variant}|${sel.quant}`, `${sel.hw}|${sel.quant}`, sel.quant, sel.hw, "default"];
    for (const k of keys) {
      const hit = config.modelNames[k];
      if (hit) return hit;
    }
    return "";
  };
  const interpolate = (text, env, modelName) => text.replace(/{{(\w+)}}/g, (_, key) => key === "MODEL_NAME" ? modelName : env[key] ?? `{{${key}}}`);
  const parseNnodes = id => {
    if (id === "single") return 1;
    const m = (/^multi-(\d+)$/).exec(id);
    return m ? parseInt(m[1], 10) : 1;
  };
  const placeholderDefaults = schema => {
    const out = {};
    for (const [k, v] of Object.entries(schema || ({}))) out[k] = v.default ?? "";
    return out;
  };
  const matchConstraint = (base, constraint) => {
    if (!constraint || typeof constraint !== "object") return false;
    const entries = Object.entries(constraint);
    if (entries.length === 0) return false;
    return entries.every(([k, vs]) => Array.isArray(vs) && vs.includes(base[k]));
  };
  const resolveRouter = (fc, sel) => {
    if (!fc || !fc.router) return null;
    const hit = (fc.routerOverrides || []).find(r => r && matchConstraint(sel, r.when));
    return hit ? {
      ...fc.router,
      ...hit
    } : fc.router;
  };
  const evaluateChip = (entry, base) => {
    if (entry === null || typeof entry !== "object") {
      return {
        value: entry,
        label: undefined,
        hidden: false,
        disabled: false,
        disableReason: ""
      };
    }
    const hidden = entry.hide ? matchConstraint(base, entry.hide) : false;
    let disabled = entry.disabled === true || entry.disable === true;
    let disableReason = typeof entry.disableReason === "function" ? entry.disableReason(base) : entry.disableReason || "";
    if (!disabled && entry.disable && typeof entry.disable === "object") {
      if (Array.isArray(entry.disable)) {
        for (const item of entry.disable) {
          const cond = item && item.when || item;
          if (matchConstraint(base, cond)) {
            disabled = true;
            if (item && item.reason) disableReason = item.reason;
            break;
          }
        }
      } else {
        disabled = matchConstraint(base, entry.disable);
      }
    }
    return {
      ...entry,
      value: entry.id !== undefined ? entry.id : entry.value,
      label: entry.label,
      hidden,
      disabled,
      disableReason
    };
  };
  const findEntry = (entries, picked) => {
    for (const e of entries || []) {
      const v = e === null || typeof e !== "object" ? e : e.id !== undefined ? e.id : e.value;
      if (v === picked) return e;
    }
    return null;
  };
  const isHidden = (entries, picked, base) => {
    const e = findEntry(entries, picked);
    if (e === null || e === undefined) return false;
    return evaluateChip(e, base).hidden;
  };
  const stripFlagsByFirstToken = (flags, prefixes) => {
    const set = new Set(prefixes);
    return flags.filter(f => !set.has(f.split(/[\s=]/)[0]));
  };
  const stripEnvByPrefix = (envList, prefixes) => {
    if (!prefixes || !prefixes.length) return envList;
    const set = new Set(prefixes);
    return envList.filter(e => !set.has(e.split("=")[0]));
  };
  const insertBeforeTail = (flags, additions) => {
    const idx = flags.findIndex(f => f.startsWith("--host"));
    const at = idx === -1 ? flags.length : idx;
    const out = flags.slice();
    out.splice(at, 0, ...additions);
    return out;
  };
  const insertAfter = (flags, afterAnyOf, additions) => {
    let idx = -1;
    for (const anchor of afterAnyOf) {
      idx = flags.findIndex(f => f.split(/[\s=]/)[0] === anchor);
      if (idx !== -1) break;
    }
    if (idx === -1) idx = flags.findIndex(f => f.startsWith("--model-path"));
    const out = flags.slice();
    out.splice(idx + 1, 0, ...additions);
    return out;
  };
  const parseIntFlag = (flags, prefix) => {
    for (const f of flags || []) {
      if (f.split(/[\s=]/)[0] !== prefix) continue;
      const rest = f.slice(prefix.length).replace(/^[\s=]+/, "");
      const n = parseInt(rest, 10);
      if (!isNaN(n)) return n;
    }
    return null;
  };
  const hasFlag = (flags, name) => (flags || []).some(f => f.split(/[\s=]/)[0] === name);
  const findFlagArg = (flags, prefix) => {
    for (const f of flags || []) {
      if (f.split(/[\s=]/)[0] !== prefix) continue;
      const rest = f.slice(prefix.length).replace(/^[\s=]+/, "");
      return rest.length ? rest : null;
    }
    return null;
  };
  const TP_HEADS = ["--tp-size", "--tp", "--tensor-parallel-size"];
  const EP_HEADS = ["--ep-size", "--ep", "--expert-parallel-size"];
  const DP_HEADS = ["--dp-size", "--dp", "--data-parallel-size"];
  const parseIntFlagAny = (flags, heads) => {
    for (const head of heads) {
      const n = parseIntFlag(flags, head);
      if (n !== null) return n;
    }
    return null;
  };
  const flagSpelling = (flags, heads, fallback) => heads.find(head => (flags || []).some(f => f.split(/[\s=]/)[0] === head)) || fallback;
  const ANCHOR_NEAR_MODEL_PATH = ["--model-path"];
  const ANCHOR_NEAR_TP = ["--tp-size", "--tp", "--model-path"];
  const ANCHOR_NEAR_DP = ["--dp-size", "--dp", "--tp-size", "--tp", "--model-path"];
  const ANCHOR_NEAR_DPATTN = ["--enable-dp-attention", "--dp-size", "--dp", "--tp-size", "--tp", "--model-path"];
  const ANCHOR_NEAR_MOE = ["--moe-a2a-backend", "--moe-runner-backend", "--enable-dp-attention", "--dp-size", "--dp", "--tp-size", "--tp", "--model-path"];
  const helpers = {
    matchConstraint,
    evaluateChip,
    findEntry,
    isHidden,
    stripFlagsByFirstToken,
    stripEnvByPrefix,
    insertBeforeTail,
    insertAfter,
    parseIntFlag,
    hasFlag,
    findFlagArg,
    TP_HEADS,
    EP_HEADS,
    DP_HEADS,
    parseIntFlagAny,
    flagSpelling,
    ANCHOR_NEAR_MODEL_PATH,
    ANCHOR_NEAR_TP,
    ANCHOR_NEAR_DP,
    ANCHOR_NEAR_DPATTN,
    ANCHOR_NEAR_MOE
  };
  const HICACHE_HEADS = ["--enable-hierarchical-cache", "--hicache-ratio", "--hicache-size", "--hicache-write-policy", "--hicache-mem-layout", "--hicache-io-backend", "--hicache-storage-backend", "--hicache-storage-prefetch-policy", "--hicache-storage-backend-extra-config"];
  const umbpRoleOverride = (fc, sel, h) => (fc.roleOverrides || []).find(r => r && sel && r.mode === sel.pdMode && (!r.when || h.matchConstraint(sel, r.when)));
  const CP_ENABLE_HEADS = ["--enable-prefill-cp", "--enable-nsa-prefill-context-parallel", "--enable-dsa-prefill-context-parallel", "--enable-prefill-context-parallel"];
  const CP_MODE_HEADS = ["--nsa-prefill-cp-mode", "--dsa-prefill-cp-mode", "--prefill-cp-mode"];
  const CP_OWNED_HEADS = [...CP_ENABLE_HEADS, ...CP_MODE_HEADS, "--cp-strategy", "--attn-cp-size"];
  const CP_MODE_TO_STRATEGY = {
    "in-seq-split": "zigzag",
    "round-robin-split": "interleave"
  };
  const cpEnabledIn = flags => CP_ENABLE_HEADS.some(head => hasFlag(flags, head));
  const bakedCpStrategy = flags => findFlagArg(flags, "--cp-strategy") || CP_MODE_TO_STRATEGY[findFlagArg(flags, "--nsa-prefill-cp-mode")] || CP_MODE_TO_STRATEGY[findFlagArg(flags, "--dsa-prefill-cp-mode")] || CP_MODE_TO_STRATEGY[findFlagArg(flags, "--prefill-cp-mode")] || null;
  const AXIS_HANDLERS = {
    attention: {
      initState: () => ({
        tp: null,
        cp: null,
        cpStrategy: null,
        dpAttn: null
      }),
      deriveFromBase: (cell, fc, h) => {
        const flags = cell && cell.flags || [];
        const dpVal = h.parseIntFlagAny(flags, h.DP_HEADS);
        const hasDpAttn = h.hasFlag(flags, "--enable-dp-attention");
        let dpAttn;
        if (dpVal !== null) dpAttn = dpVal; else if (hasDpAttn) dpAttn = 1; else dpAttn = false;
        const cpSize = h.parseIntFlag(flags, "--attn-cp-size");
        return {
          tp: h.parseIntFlagAny(flags, h.TP_HEADS),
          cp: cpEnabledIn(flags) ? cpSize !== null ? cpSize : 2 : null,
          cpStrategy: bakedCpStrategy(flags),
          dpAttn
        };
      },
      apply: ({flags, env, value, fc, sel, h}) => {
        const knobEntry = id => (fc.knobs || []).find(k => k.id === id) || ({});
        const factsNow = () => ({
          ...sel || ({}),
          dpAttnOn: h.hasFlag(flags, "--enable-dp-attention"),
          cpOn: cpEnabledIn(flags),
          cpStrategy: bakedCpStrategy(flags) || "interleave",
          effTp: h.parseIntFlagAny(flags, h.TP_HEADS)
        });
        const cpSizeTargetNow = () => {
          if (knobEntry("cp").freeSize) return null;
          const dpIntent = value.dpAttn !== null && value.dpAttn !== undefined ? value.dpAttn : h.hasFlag(flags, "--enable-dp-attention") ? h.parseIntFlagAny(flags, h.DP_HEADS) ?? 1 : false;
          if (typeof dpIntent === "number" && dpIntent > 1) return null;
          return h.parseIntFlagAny(flags, h.TP_HEADS);
        };
        const blocked = (id, v) => {
          const facts = factsNow();
          const kc = h.evaluateChip(knobEntry(id), facts);
          if (kc.hidden || kc.disabled) return true;
          if (id === "cp" && typeof v === "number" && v > 1) {
            const target = cpSizeTargetNow();
            if (target !== null && v !== target) return true;
          }
          const e = h.findEntry(knobEntry(id).values || [], v);
          return !!(e !== null && e !== undefined && h.evaluateChip(e, facts).disabled);
        };
        if (value.tp !== null && !blocked("tp", value.tp)) {
          const tpHead = h.flagSpelling(flags, h.TP_HEADS, "--tp");
          flags = h.stripFlagsByFirstToken(flags, h.TP_HEADS);
          flags = h.insertAfter(flags, h.ANCHOR_NEAR_MODEL_PATH, [`${tpHead} ${value.tp}`]);
        }
        const cpStrategyOverride = value.cpStrategy && !blocked("cpStrategy", value.cpStrategy) ? value.cpStrategy : null;
        const cpPick = value.cp !== null ? value.cp : cpStrategyOverride && cpEnabledIn(flags) ? h.parseIntFlag(flags, "--attn-cp-size") ?? 2 : null;
        const cpStrategyPick = cpStrategyOverride || bakedCpStrategy(flags) || "interleave";
        if (cpPick !== null && !blocked("cp", cpPick)) {
          flags = h.stripFlagsByFirstToken(flags, CP_OWNED_HEADS);
          if (cpPick > 1) {
            flags = h.insertAfter(flags, h.ANCHOR_NEAR_DPATTN, [`--attn-cp-size ${cpPick}`, "--enable-prefill-cp", `--cp-strategy ${cpStrategyPick}`]);
          }
        }
        const dpForced = (knobEntry("dpAttn").forceOff || []).find(r => r && h.matchConstraint(factsNow(), r.when));
        if (dpForced) {
          flags = h.stripFlagsByFirstToken(flags, [...h.DP_HEADS, "--enable-dp-attention", "--enable-dp-attention-local-control-broadcast"]);
          const stripEnv = dpForced.stripEnv || [];
          if (stripEnv.length) {
            env = env.filter(e => !stripEnv.includes(e.split("=")[0]));
          }
        } else if (value.dpAttn !== null && value.dpAttn !== undefined && !blocked("dpAttn", value.dpAttn)) {
          const dpHead = h.flagSpelling(flags, h.DP_HEADS, "--dp-size");
          const hadLocalBroadcast = h.hasFlag(flags, "--enable-dp-attention-local-control-broadcast");
          flags = h.stripFlagsByFirstToken(flags, [...h.DP_HEADS, "--enable-dp-attention", "--enable-dp-attention-local-control-broadcast"]);
          if (typeof value.dpAttn === "number" && value.dpAttn > 0) {
            flags = h.insertAfter(flags, h.ANCHOR_NEAR_TP, [`${dpHead} ${value.dpAttn}`, "--enable-dp-attention", ...hadLocalBroadcast ? ["--enable-dp-attention-local-control-broadcast"] : []]);
          }
        }
        return {
          flags,
          env
        };
      },
      render: ({axisId, value, setValue, fc, base, s, h, renderSelect, derived}) => {
        const knobs = fc.knobs || [];
        if (!knobs.length) return null;
        const setKnob = (k, v) => setValue({
          ...value,
          [k]: v
        });
        const labelFor = knob => c => {
          if (c.label !== undefined) return c.label;
          if (knob.id === "dpAttn") {
            const labelMap = knob.labels || ({
              "auto": "Auto",
              "false": "Off"
            });
            const k = c.value === null ? "auto" : String(c.value);
            return labelMap[k] || k;
          }
          return c.value === null ? "Auto" : String(c.value);
        };
        const knobDisplay = knob => {
          const v = value[knob.id];
          if (v !== null && v !== undefined) return v;
          if (derived && derived[knob.id] !== undefined) return derived[knob.id];
          return null;
        };
        const hideNullFor = knob => {
          const d = derived ? derived[knob.id] : null;
          return d !== null && d !== undefined ? [null] : [];
        };
        const entriesFor = knob => {
          const vals = knob.values || [null];
          if (knob.id !== "cp" || knob.freeSize) return vals;
          const target = base.cpSizeTarget;
          if (target === null || target === undefined) return vals;
          return vals.map(entry => {
            const v = entry === null || typeof entry !== "object" ? entry : entry.id !== undefined ? entry.id : entry.value;
            if (typeof v !== "number" || v <= 1 || v === target) return entry;
            const wrapped = entry === null || typeof entry !== "object" ? {
              value: entry
            } : {
              ...entry
            };
            return {
              ...wrapped,
              disabled: true,
              disableReason: `SGLang derives the prefill-CP size as attn_cp_size = TP / DP-Attention (= ${target} here), so only that size can be enabled.`
            };
          });
        };
        return <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>Attention</span>
              {knobs.map(knob => {
          const kc = h.evaluateChip(knob, base);
          if (kc.hidden) return null;
          const forced = (knob.forceOff || []).find(r => r && h.matchConstraint(base, r.when));
          return <span key={knob.id} style={s.field}>
                    <span style={s.fieldLabel}>{knob.label || knob.id.toUpperCase()}</span>
                    {renderSelect(forced ? false : knobDisplay(knob), entriesFor(knob), nv => setKnob(knob.id, nv), base, labelFor(knob), {
            hideValues: hideNullFor(knob),
            disabled: kc.disabled || !!forced,
            disabledReason: forced ? forced.reason : kc.disableReason
          })}
                  </span>;
        })}
            </div>
          </div>;
      }
    },
    moe: {
      initState: () => ({
        backend: null,
        ep: null,
        mmQuant: null
      }),
      deriveFromBase: (cell, fc, h) => {
        const flags = cell && cell.flags || [];
        const a2a = h.findFlagArg(flags, "--moe-a2a-backend");
        const runner = h.findFlagArg(flags, "--moe-runner-backend");
        const w4a4 = h.hasFlag(flags, "--enable-w4a4-mxfp4-megamoe");
        return {
          backend: a2a || runner || null,
          ep: h.parseIntFlagAny(flags, h.EP_HEADS),
          mmQuant: w4a4 ? "w4a4" : "w4a8"
        };
      },
      apply: ({flags, env, value, fc, h, derived}) => {
        if (value.backend !== null) {
          flags = h.stripFlagsByFirstToken(flags, ["--moe-a2a-backend", "--moe-runner-backend"]);
          const backendEnvKeys = [];
          for (const o of fc.backend?.options || []) {
            for (const e of o.env || []) backendEnvKeys.push(e.split("=")[0]);
          }
          if (backendEnvKeys.length) env = h.stripEnvByPrefix(env, backendEnvKeys);
          const opt = (fc.backend?.options || []).find(o => o.id === value.backend);
          if (opt?.flags?.length) {
            flags = h.insertAfter(flags, h.ANCHOR_NEAR_DPATTN, opt.flags);
          }
          if (opt?.env?.length) env = [...env, ...opt.env];
        }
        const mq = fc.megamoeQuant;
        if (mq) {
          const quantKeys = [];
          const quantFlagHeads = [];
          for (const o of mq.options || []) {
            for (const e of o.env || []) quantKeys.push(e.split("=")[0]);
            for (const f of o.flags || []) quantFlagHeads.push(f.split(/[\s=]/)[0]);
          }
          flags = h.stripFlagsByFirstToken(flags, quantFlagHeads);
          const effBackend = value.backend !== null ? value.backend : derived && derived.backend;
          if (effBackend === "megamoe") {
            env = h.stripEnvByPrefix(env, [...mq.stripEnv || [], ...quantKeys]);
            const quant = value.mmQuant != null ? value.mmQuant : derived && derived.mmQuant || "w4a8";
            const opt = (mq.options || []).find(o => o.id === quant);
            if (opt?.flags?.length) {
              flags = h.insertAfter(flags, h.ANCHOR_NEAR_MOE, opt.flags);
            }
            if (opt?.env?.length) env = [...env, ...opt.env];
          } else if (value.backend !== null) {
            env = h.stripEnvByPrefix(env, quantKeys);
          }
        }
        if (value.ep !== null) {
          const epHead = h.flagSpelling(flags, h.EP_HEADS, "--ep");
          flags = h.stripFlagsByFirstToken(flags, h.EP_HEADS);
          if (value.ep > 1) {
            flags = h.insertAfter(flags, h.ANCHOR_NEAR_MOE, [`${epHead} ${value.ep}`]);
          }
        }
        return {
          flags,
          env
        };
      },
      render: ({axisId, value, setValue, fc, base, s, renderSelect, derived}) => {
        if (!fc.backend && !fc.ep) return null;
        const setSlot = (k, v) => setValue({
          ...value,
          [k]: v
        });
        const slotDisplay = k => {
          const v = value[k];
          if (v !== null && v !== undefined) return v;
          if (derived && derived[k] !== undefined) return derived[k];
          return null;
        };
        const hideNull = k => {
          const d = derived ? derived[k] : null;
          return d !== null && d !== undefined ? [null] : [];
        };
        const mmOpt = (fc.backend?.options || []).find(o => o.id === "megamoe");
        const mmAvail = !!mmOpt && (!mmOpt.requiresHw || mmOpt.requiresHw.includes(base.hw)) && (!mmOpt.excludesStrategy || !mmOpt.excludesStrategy.includes(base.strategy));
        const backendIsMega = slotDisplay("backend") === "megamoe";
        const epShown = !!fc.ep && !(typeof fc.ep.showWhen === "function" && !fc.ep.showWhen(base));
        return <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>MoE</span>
              {fc.backend && <span style={s.field}>
                  <span style={s.fieldLabel}>Backend</span>
                  {renderSelect(slotDisplay("backend"), fc.backend.options || [], v => setSlot("backend", v), base, undefined, {
          hideValues: [...hideNull("backend"), ...mmAvail ? [] : ["megamoe"]]
        })}
                </span>}
              {fc.megamoeQuant && backendIsMega && <span style={s.field}>
                  <span style={s.fieldLabel}>Quantization</span>
                  {renderSelect(value.mmQuant != null ? value.mmQuant : derived && derived.mmQuant || "w4a8", fc.megamoeQuant.options || [], v => setSlot("mmQuant", v), base)}
                </span>}
              {epShown && <span style={s.field}>
                  <span style={s.fieldLabel}>{fc.ep.label || "EP"}</span>
                  {renderSelect(slotDisplay("ep"), fc.ep.values || [null], v => setSlot("ep", v), base, undefined, {
          hideValues: hideNull("ep")
        })}
                </span>}
            </div>
          </div>;
      }
    },
    parsers: {
      initState: fc => {
        const out = {};
        for (const item of fc.items || []) out[item.id] = null;
        return out;
      },
      deriveFromBase: (cell, fc, h) => {
        const flags = cell && cell.flags || [];
        const out = {};
        for (const item of fc.items || []) {
          const prefix = item.flag.split(/[\s=]/)[0];
          out[item.id] = h.hasFlag(flags, prefix);
        }
        return out;
      },
      apply: ({flags, env, value, fc, h, derived}) => {
        const items = fc.items || [];
        const eff = {};
        const baseOf = {};
        for (const item of items) {
          baseOf[item.id] = derived ? !!derived[item.id] : false;
          const v = value[item.id];
          eff[item.id] = v === null || v === undefined ? baseOf[item.id] : v;
        }
        const anyOverride = items.some(it => eff[it.id] !== baseOf[it.id]);
        if (!anyOverride) return {
          flags,
          env
        };
        flags = h.stripFlagsByFirstToken(flags, ["--reasoning-parser", "--tool-call-parser"]);
        const adds = [];
        for (const item of items) {
          if (eff[item.id]) adds.push(item.flag);
        }
        if (adds.length) flags = h.insertBeforeTail(flags, adds);
        return {
          flags,
          env
        };
      },
      render: ({axisId, value, setValue, fc, base, s, h, renderChip, derived}) => {
        const visible = (fc.items || []).map(item => ({
          item,
          c: h.evaluateChip(item, base)
        })).filter(({c}) => !c.hidden);
        if (visible.length === 0) return null;
        const effOn = id => {
          const v = value[id];
          if (v !== null && v !== undefined) return v;
          if (derived && derived[id] !== undefined) return derived[id];
          return false;
        };
        return <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>Parsers</span>
              {visible.map(({item, c}) => <span key={item.id} style={s.field}>
                  {renderChip(item.label, effOn(item.id), true, () => setValue({
          ...value,
          [item.id]: !effOn(item.id)
        }), {
          disabled: c.disabled,
          disabledReason: c.disableReason
        })}
                </span>)}
            </div>
          </div>;
      }
    },
    speculative: {
      initState: () => "current",
      deriveFromBase: (cell, fc) => {
        const flags = cell && cell.flags || [];
        const baseSpec = flags.filter(f => {
          const head = f.split(/[\s=]/)[0];
          return head === "--speculative-algorithm" || head === "--speculative-num-steps" || head === "--speculative-eagle-topk" || head === "--speculative-num-draft-tokens" || head === "--speculative-adaptive" || head === "--speculative-dspark-block-size" || head === "--enable-linear-replayssm-spec" || head === "--linear-replayssm-cache-len" || head === "--speculative-ngram-max-bfs-breadth";
        });
        if (baseSpec.length === 0) return "off";
        for (const opt of fc.options || []) {
          if (!opt.flags || opt.flags.length !== baseSpec.length) continue;
          const ok = opt.flags.every(pf => baseSpec.includes(pf));
          if (ok) return opt.id;
        }
        return "current";
      },
      apply: ({flags, env, value, fc, sel, h, derived}) => {
        if (value === "current") return {
          flags,
          env
        };
        if (derived && value === derived) return {
          flags,
          env
        };
        const picked = (fc.options || []).find(p => p.id === value);
        if (picked && h.evaluateChip(picked, {
          ...sel,
          dpAttnOn: h.hasFlag(flags, "--enable-dp-attention")
        }).disabled) {
          return {
            flags,
            env
          };
        }
        flags = h.stripFlagsByFirstToken(flags, ["--speculative-algorithm", "--speculative-num-steps", "--speculative-eagle-topk", "--speculative-num-draft-tokens", "--speculative-adaptive", "--speculative-dspark-block-size", "--enable-linear-replayssm-spec", "--linear-replayssm-cache-len", "--speculative-ngram-max-bfs-breadth"]);
        const preset = (fc.options || []).find(p => p.id === value);
        if (preset?.flags?.length) flags = h.insertBeforeTail(flags, preset.flags);
        return {
          flags,
          env
        };
      },
      render: ({axisId, value, setValue, fc, base, s, h, renderChip, derived}) => {
        const opts = fc.options || [];
        if (!opts.length) return null;
        const display = value !== "current" ? value : derived ? derived : "current";
        const hideCurrent = !!(derived && derived !== "current");
        const visible = opts.map(opt => h.evaluateChip(opt, base)).filter(c => !c.hidden && !(hideCurrent && c.value === "current"));
        if (visible.length === 0) return null;
        const note = (visible.find(c => c.value === display) || ({})).note;
        return <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>Speculative</span>
              {visible.map(c => <span key={c.value} style={s.field}>
                  {renderChip(c.label, display, c.value, () => setValue(c.value), {
          disabled: c.disabled,
          disabledReason: c.disableReason
        })}
                </span>)}
            </div>
            {note && <div style={s.axisNote}>{note}</div>}
          </div>;
      }
    },
    pdDisagg: {
      initState: (fc, base) => {
        const pick = entries => (entries || []).find(e => e && e.defaultWhen && base && matchConstraint(base, e.defaultWhen));
        const backends = fc && fc.transferBackends || [];
        return {
          mode: "off",
          transferBackend: (pick(backends) || backends[0] || ({})).id || "mooncake",
          ibDevice: (pick(fc && fc.ibDevices) || ({})).id || "auto"
        };
      },
      apply: ({flags, env, value, sel, fc, h}) => {
        const bootstrapPort = h.findFlagArg(flags, "--disaggregation-bootstrap-port");
        flags = h.stripFlagsByFirstToken(flags, ["--disaggregation-mode", "--disaggregation-transfer-backend", "--disaggregation-ib-device", "--disaggregation-bootstrap-port"]);
        const backends = fc.transferBackends || [];
        const mode = (fc.modes || []).length ? value.mode : sel && sel.pdMode || "off";
        if (mode === "prefill" || mode === "decode") {
          const specAlgorithm = (h.findFlagArg(flags, "--speculative-algorithm") || "").toUpperCase();
          if ((fc.incompatibleSpeculativeAlgorithms || []).includes(specAlgorithm)) {
            flags = flags.filter(flag => !flag.split(/[\s=]/)[0].startsWith("--speculative-"));
          }
          const backend = value.transferBackend || (backends[0] || ({})).id || "mooncake";
          const adds = [`--disaggregation-mode ${mode}`, `--disaggregation-transfer-backend ${backend}`];
          if (bootstrapPort) {
            adds.push(`--disaggregation-bootstrap-port ${bootstrapPort}`);
          }
          if (value.ibDevice && value.ibDevice !== "auto") {
            adds.push(`--disaggregation-ib-device ${value.ibDevice}`);
          }
          const modeMeta = (fc.modes || []).find(m => m.id === mode);
          const roleOverride = (fc.roleOverrides || []).find(r => r && r.mode === mode && r.when && h.matchConstraint(sel, r.when));
          const roleSpec = roleOverride || modeMeta;
          const modeGate = roleOverride ? null : modeMeta && modeMeta.when;
          const modeOk = !modeGate || Object.keys(modeGate).every(k => (modeGate[k] || []).includes(sel[k]));
          if (modeOk && roleSpec && roleSpec.flags && roleSpec.flags.length) {
            flags = h.stripFlagsByFirstToken(flags, roleSpec.flags.map(f => f.split(/[\s=]/)[0]));
            adds.push(...roleSpec.flags);
          }
          if (modeOk && roleSpec && roleSpec.stripFlags && roleSpec.stripFlags.length) {
            flags = h.stripFlagsByFirstToken(flags, roleSpec.stripFlags);
          }
          flags = h.insertBeforeTail(flags, adds);
          const servePort = PD_PORTS[mode].serve;
          flags = flags.map(f => f.split(/[\s=]/)[0] === "--port" ? `--port ${servePort}` : f);
          const meta = backends.find(b => b.id === backend);
          if (meta && meta.env && meta.env.length) {
            const gate = meta.envWhen;
            const ok = !gate || Object.keys(gate).every(k => (gate[k] || []).includes(sel[k]));
            if (ok) env = [...env, ...meta.env.filter(e => !env.includes(e))];
          }
          if (modeOk && roleSpec && roleSpec.env && roleSpec.env.length) {
            env = [...env, ...roleSpec.env.filter(e => !env.includes(e))];
          }
        }
        return {
          flags,
          env
        };
      },
      getRenderHints: (value, fc, context) => {
        const specAlgorithm = (context.h.findFlagArg(context.flags, "--speculative-algorithm") || "").toUpperCase();
        if ((fc.incompatibleSpeculativeAlgorithms || []).includes(specAlgorithm)) {
          return null;
        }
        if (value.mode === "prefill" || value.mode === "decode") {
          return {
            pdMode: value.mode
          };
        }
        return null;
      },
      render: ({axisId, value, setValue, fc, base, s, h, renderSelect}) => {
        const setSlot = (k, v) => setValue({
          ...value,
          [k]: v
        });
        const showModes = (fc.modes || []).length > 0;
        const showBackends = (fc.transferBackends || []).length > 0;
        const showIb = (fc.ibDevices || []).length > 0;
        if (!showModes && !showBackends && !showIb) return null;
        const note = (fc.notes || []).find(n => n && (!n.mode || [].concat(n.mode).includes(value.mode)) && h.matchConstraint(base, n.when));
        return <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>PD Disagg</span>
              {showModes && <span style={s.field}>
                  <span style={s.fieldLabel}>Mode</span>
                  {renderSelect(value.mode, fc.modes, v => setSlot("mode", v), base)}
                </span>}
              {showBackends && <span style={s.field}>
                  <span style={s.fieldLabel}>Transfer Backend</span>
                  {renderSelect(value.transferBackend, fc.transferBackends, v => setSlot("transferBackend", v), base)}
                </span>}
              {showIb && <span style={s.field}>
                  <span style={s.fieldLabel}>IB Device</span>
                  {renderSelect(value.ibDevice, fc.ibDevices, v => setSlot("ibDevice", v), base)}
                </span>}
            </div>
            {note && <div style={s.axisNote}>{note.text}</div>}
          </div>;
      }
    },
    hisparse: {
      initState: fc => ({
        enable: false,
        hostRatio: fc && fc.defaultHostRatio || null
      }),
      apply: ({flags, env, value, fc, h}) => {
        const ownedHeads = ["--enable-hisparse", "--hisparse-config", ...(fc.requiredFlags || []).map(f => f.split(/\s/)[0])];
        flags = h.stripFlagsByFirstToken(flags, ownedHeads);
        const isDecode = flags.includes("--disaggregation-mode decode");
        if (value.enable && isDecode) {
          const ratio = value.hostRatio !== null && value.hostRatio !== undefined ? value.hostRatio : fc.defaultHostRatio || 10;
          const cfg = {
            ...fc.config || ({}),
            host_to_device_ratio: ratio
          };
          const adds = [...fc.requiredFlags || [], "--enable-hisparse", `--hisparse-config '${JSON.stringify(cfg)}'`];
          flags = h.insertBeforeTail(flags, adds);
        }
        return {
          flags,
          env
        };
      },
      render: ({axisId, value, setValue, fc, base, s, renderChip, renderSelect}) => {
        if (base.pdMode !== "decode") return null;
        const setSlot = (k, v) => setValue({
          ...value,
          [k]: v
        });
        const hasRatios = (fc.hostRatios || []).length > 0;
        return <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>HiSparse</span>
              {typeof fc.showWhen !== "function" && <span style={s.field}>
                  {renderChip("Enable", value.enable, true, () => setSlot("enable", !value.enable))}
                </span>}
              {hasRatios && <span style={s.field}>
                  <span style={s.fieldLabel}>Host ratio</span>
                  {renderSelect(value.hostRatio, fc.hostRatios, v => setSlot("hostRatio", v), base)}
                </span>}
            </div>
          </div>;
      }
    },
    hicache: {
      initState: () => ({
        enable: null,
        backend: null,
        writePolicy: "auto"
      }),
      deriveFromBase: (cell, fc, h) => {
        const flags = cell && cell.flags || [];
        return {
          enable: h.hasFlag(flags, "--enable-hierarchical-cache"),
          backend: h.findFlagArg(flags, "--hicache-storage-backend"),
          writePolicy: h.findFlagArg(flags, "--hicache-write-policy") || "auto"
        };
      },
      apply: ({flags, env, value, fc, sel, h, derived}) => {
        if (fc.excludesHw && sel && fc.excludesHw.includes(sel.hw)) return {
          flags,
          env
        };
        if (typeof fc.showWhen === "function") {
          const set = (name, val) => {
            flags = h.stripFlagsByFirstToken(flags, [name]);
            if (val) flags = h.insertBeforeTail(flags, [`${name} ${val}`]);
          };
          if (value.backend) set("--hicache-storage-backend", value.backend);
          if (value.writePolicy && value.writePolicy !== "auto") {
            set("--hicache-write-policy", value.writePolicy);
          }
          return {
            flags,
            env
          };
        }
        const hasOverride = value.enable !== null || value.backend !== null || value.writePolicy && value.writePolicy !== "auto";
        if (!hasOverride) return {
          flags,
          env
        };
        const backendOptions = fc.backends || [];
        const ownedHeads = [...HICACHE_HEADS, ...(fc.requiredFlags || []).map(f => f.split(/\s/)[0]), ...backendOptions.flatMap(o => (o.flags || []).map(f => f.split(/\s/)[0]))];
        const ownedEnvKeys = [...fc.requiredEnv || [], ...backendOptions.flatMap(o => o.env || [])].map(e => e.split("=")[0]);
        flags = h.stripFlagsByFirstToken(flags, ownedHeads);
        if (ownedEnvKeys.length) env = h.stripEnvByPrefix(env, ownedEnvKeys);
        const enabled = value.enable !== null ? value.enable : !!(derived && derived.enable);
        const backend = value.backend !== null ? value.backend : derived && derived.backend || fc.defaultBackend || null;
        if (enabled) {
          const isAmd = sel && (/^mi\d/).test(sel.hw);
          const pdMode = h.findFlagArg(flags, "--disaggregation-mode") || "off";
          const pdBackend = h.findFlagArg(flags, "--disaggregation-transfer-backend");
          const roleOverride = (fc.roleOverrides || []).find(item => {
            if (!item || item.mode !== pdMode) return false;
            if (item.transferBackend && item.transferBackend !== pdBackend) return false;
            return !item.when || h.matchConstraint(sel, item.when);
          });
          const amdIo = roleOverride || isAmd && fc.amdIo;
          const ratio = amdIo && amdIo.ratio || 2;
          const useAmdIo = isAmd && amdIo;
          const adds = ["--enable-hierarchical-cache", `--hicache-ratio ${ratio}`];
          if (!useAmdIo) {
            adds.push("--hicache-size 0");
          }
          if (useAmdIo) {
            adds.push(`--hicache-mem-layout ${amdIo.memLayout}`, `--hicache-io-backend ${amdIo.ioBackend}`);
          } else if (backend) {
            adds.push("--hicache-mem-layout page_first_direct", "--hicache-io-backend direct");
          }
          const writePolicy = value.writePolicy && value.writePolicy !== "auto" ? value.writePolicy : amdIo && amdIo.writePolicy || "write_through";
          adds.push(`--hicache-write-policy ${writePolicy}`);
          if (isAmd && fc.amdStorageFileOnly ? backend === "file" : !!backend) {
            adds.push(`--hicache-storage-backend ${backend}`, `--hicache-storage-prefetch-policy ${amdIo && amdIo.prefetchPolicy || "wait_complete"}`);
          } else if (amdIo && amdIo.prefetchPolicy) {
            adds.push(`--hicache-storage-prefetch-policy ${amdIo.prefetchPolicy}`);
          }
          const backendOption = backendOptions.find(o => o.id === backend);
          adds.push(...backendOption?.flags || [], ...fc.requiredFlags || []);
          flags = h.insertBeforeTail(flags, adds);
          env = [...env, ...backendOption?.env || [], ...fc.requiredEnv || []];
        }
        return {
          flags,
          env
        };
      },
      render: ({axisId, value, setValue, fc, base, s, renderChip, renderSelect, derived}) => {
        if (fc.excludesHw && fc.excludesHw.includes(base.hw)) return null;
        const setSlot = (k, v) => setValue({
          ...value,
          [k]: v
        });
        const hasBackends = (fc.backends || []).length > 0;
        const hasPolicies = (fc.writePolicies || []).length > 0;
        const enabled = value.enable !== null ? value.enable : !!(derived && derived.enable);
        const hasAutoBackend = (fc.backends || []).some(o => o.id === null);
        const backend = value.backend !== null ? value.backend : hasAutoBackend ? null : derived && derived.backend || fc.defaultBackend || null;
        const writePolicy = value.writePolicy !== "auto" ? value.writePolicy : derived && derived.writePolicy || "auto";
        return <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>HiCache</span>
              {typeof fc.showWhen !== "function" && <span style={s.field}>
                  {renderChip("Enable", enabled, true, () => setSlot("enable", !enabled))}
                </span>}
              {hasBackends && <span style={s.field}>
                  <span style={s.fieldLabel}>Storage</span>
                  {renderSelect(backend, fc.backends, v => setSlot("backend", v), base)}
                </span>}
              {hasPolicies && <span style={s.field}>
                  <span style={s.fieldLabel}>Write</span>
                  {renderSelect(writePolicy, fc.writePolicies, v => setSlot("writePolicy", v), base)}
                </span>}
            </div>
          </div>;
      }
    },
    umbp: {
      initState: () => ({
        enable: null,
        backend: null
      }),
      deriveFromBase: (cell, fc, h) => {
        const flags = cell && cell.flags || [];
        return {
          enable: h.hasFlag(flags, "--enable-unified-cache-external-linker"),
          backend: h.findFlagArg(flags, "--unified-cache-external-linker-backend")
        };
      },
      apply: ({flags, env, value, fc, sel, h, derived}) => {
        const roleOverride = umbpRoleOverride(fc, sel, h);
        const ownedHeads = ["--enable-unified-cache-external-linker", "--unified-cache-external-linker-backend", ...(fc.requiredFlags || []).map(f => f.split(/\s/)[0])];
        flags = h.stripFlagsByFirstToken(flags, ownedHeads);
        const ownedEnv = [...fc.requiredEnv || [], ...roleOverride && roleOverride.env || []];
        if (ownedEnv.length) {
          env = h.stripEnvByPrefix(env, ownedEnv.map(e => e.split("=")[0]));
        }
        const enabled = value.enable !== null ? value.enable : roleOverride ? !!roleOverride.enable : !!(derived && derived.enable);
        if (!enabled) return {
          flags,
          env
        };
        if (fc.onlyHw && sel && !fc.onlyHw.includes(sel.hw)) return {
          flags,
          env
        };
        if (fc.requiresDpAttention && !(roleOverride && roleOverride.allowTp) && !flags.some(f => f.split(/[\s=]/)[0] === "--enable-dp-attention")) {
          return {
            flags,
            env
          };
        }
        flags = h.stripFlagsByFirstToken(flags, HICACHE_HEADS);
        const backend = value.backend || derived && derived.backend || fc.defaultBackend || "mori";
        flags = h.insertBeforeTail(flags, ["--enable-unified-cache-external-linker", `--unified-cache-external-linker-backend ${backend}`, ...fc.requiredFlags || [], ...roleOverride && roleOverride.flags || []]);
        const extraEnv = [...fc.requiredEnv || [], ...roleOverride && roleOverride.env || []];
        env = [...env, ...extraEnv.filter(e => !env.includes(e))];
        return {
          flags,
          env
        };
      },
      render: ({axisId, value, setValue, fc, base, s, h, renderChip, renderSelect, derived}) => {
        if (fc.onlyHw && !fc.onlyHw.includes(base.hw)) return null;
        const setSlot = (k, v) => setValue({
          ...value,
          [k]: v
        });
        const roleOverride = umbpRoleOverride(fc, base, h);
        const enabled = value.enable !== null ? value.enable : roleOverride ? !!roleOverride.enable : !!(derived && derived.enable);
        const needsDp = !!fc.requiresDpAttention && !base.dpAttnOn && !(roleOverride && roleOverride.allowTp);
        const backend = value.backend !== null ? value.backend : derived && derived.backend || fc.defaultBackend || "mori";
        return <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>UMBP</span>
              <span style={s.field}>
                {renderChip("Enable", enabled, true, () => setSlot("enable", !enabled), {
          disabled: needsDp,
          disabledReason: needsDp ? "Needs DP Attention — the linker keyspace is per DP rank, so under pure TP the store holds one copy per TP rank." : ""
        })}
              </span>
              {(fc.backends || []).length > 0 && <span style={s.field}>
                  <span style={s.fieldLabel}>Store</span>
                  {renderSelect(backend, fc.backends, v => setSlot("backend", v), base)}
                </span>}
            </div>
            {enabled && !needsDp && roleOverride && roleOverride.note && <div style={s.axisNote}>{roleOverride.note}</div>}
          </div>;
      }
    },
    flagSelects: {
      initState: (fc, base) => {
        const out = {};
        for (const spec of fc || []) {
          const d = typeof spec.default === "function" ? spec.default(base) : spec.default;
          out[spec.id] = d ?? null;
        }
        return out;
      },
      deriveFromBase: (cell, fc) => {
        const flags = cell && cell.flags || [];
        const out = {};
        for (const spec of fc || []) {
          const prefixes = spec.stripPrefixes || [];
          const fam = flags.filter(f => prefixes.includes(f.split(/[\s=]/)[0]));
          let hit = null;
          for (const opt of spec.options || []) {
            if (typeof opt.flags === "function") continue;
            const of = opt.flags || [];
            if (of.length === fam.length && of.every(x => fam.includes(x))) {
              hit = opt.id;
              break;
            }
          }
          out[spec.id] = hit;
        }
        return out;
      },
      apply: ({flags, env, value, fc, sel, h, derived}) => {
        const evalBase = {
          ...sel || ({}),
          dpAttnOn: h.hasFlag(flags, "--enable-dp-attention"),
          pdMode: h.findFlagArg(flags, "--disaggregation-mode") || "off"
        };
        for (const spec of fc || []) {
          if (typeof spec.showWhen === "function" && !spec.showWhen(sel, value, derived)) continue;
          const v = value ? value[spec.id] : null;
          if (v === null || v === undefined) continue;
          const d = derived ? derived[spec.id] : null;
          if (v === d) continue;
          const opt = (spec.options || []).find(o => o.id === v);
          if (!opt) continue;
          if (h.evaluateChip(opt, evalBase).disabled) continue;
          const optFlags = typeof opt.flags === "function" ? opt.flags(value, evalBase) : opt.flags || [];
          if (optFlags === null) continue;
          const strip = new Set(spec.stripPrefixes || []);
          const byTok = new Map();
          for (const f of optFlags) {
            const t = f.split(/[\s=]/)[0];
            if (!byTok.has(t)) byTok.set(t, []);
            byTok.get(t).push(f);
          }
          const consumed = new Set();
          const next = [];
          for (const f of flags) {
            const t = f.split(/[\s=]/)[0];
            if (byTok.has(t)) {
              if (!consumed.has(t)) {
                next.push(...byTok.get(t));
                consumed.add(t);
              }
            } else if (!strip.has(t)) {
              next.push(f);
            }
          }
          const fresh = [];
          for (const [t, fs] of byTok) {
            if (!consumed.has(t)) fresh.push(...fs);
          }
          flags = fresh.length ? h.insertBeforeTail(next, fresh) : next;
          const envKeys = [...spec.stripEnv || []];
          for (const o of spec.options || []) {
            for (const e of o.env || []) envKeys.push(e.split("=")[0]);
          }
          if (envKeys.length) env = h.stripEnvByPrefix(env, envKeys);
          if (opt.env && opt.env.length) env = [...env, ...opt.env];
        }
        return {
          flags,
          env
        };
      },
      render: ({axisId, value, setValue, fc, base, s, h, renderChip, derived}) => {
        const cards = [];
        for (const spec of fc || []) {
          if (typeof spec.showWhen === "function" && !spec.showWhen(base, value, derived)) continue;
          const opts = (spec.options || []).map(o => h.evaluateChip(o, base)).filter(c => !c.hidden);
          if (!opts.length) continue;
          const explicit = value ? value[spec.id] : null;
          const display = explicit !== null && explicit !== undefined ? explicit : derived ? derived[spec.id] : null;
          if (spec.control === "slider") {
            const idx = Math.max(0, opts.findIndex(c => c.value === display));
            const cur = opts[idx];
            cards.push(<div key={`${axisId}-${spec.id}`} style={s.card}>
                <div style={s.compactRow}>
                  <span style={s.axisTitle}>{spec.title}</span>
                  <input type="range" min={0} max={opts.length - 1} step={1} value={idx} onChange={e => setValue({
              ...value,
              [spec.id]: opts[Number(e.target.value)].value
            })} style={{
              flex: 1,
              minWidth: "120px",
              accentColor: "#D45D44"
            }} />
                  <span style={{
              ...s.axisTitle,
              minWidth: "24px",
              textAlign: "right"
            }}>
                    {cur ? cur.label : "-"}
                  </span>
                </div>
              </div>);
            continue;
          }
          cards.push(<div key={`${axisId}-${spec.id}`} style={s.card}>
              <div style={s.compactRow}>
                <span style={s.axisTitle}>{spec.title}</span>
                {opts.map(c => <span key={c.value} style={s.field}>
                    {renderChip(c.label, display, c.value, () => setValue({
            ...value,
            [spec.id]: c.value
          }), {
            disabled: c.disabled,
            disabledReason: c.disableReason
          })}
                  </span>)}
              </div>
            </div>);
        }
        return cards.length ? cards : null;
      }
    }
  };
  const applyAllDeltas = (baseFlags, baseEnv, allDeltas, sel, derivedMap) => {
    let flags = [...baseFlags];
    let env = [...baseEnv || []];
    let pdMode = null;
    const pdFc = pgFeatures.pdDisagg;
    const pdDelta = allDeltas.pdDisagg;
    const pdRoleSel = pdFc && (pdFc.modes || []).length && pdDelta ? pdDelta.mode : sel && sel.pdMode || "off";
    for (const [axisId, handler] of Object.entries(AXIS_HANDLERS)) {
      const fc = pgFeatures[axisId];
      if (!fc) continue;
      const value = allDeltas[axisId];
      if (value === undefined) continue;
      const derived = derivedMap ? derivedMap[axisId] : null;
      const specAlgorithm = (findFlagArg(flags, "--speculative-algorithm") || "").toUpperCase() || null;
      const liveSel = {
        ...sel,
        specAlgorithm,
        pdMode: pdRoleSel
      };
      const out = handler.apply({
        flags,
        env,
        value,
        fc,
        sel: liveSel,
        h: helpers,
        derived
      });
      flags = out.flags;
      env = out.env;
      if (handler.getRenderHints) {
        const hints = handler.getRenderHints(value, fc, {
          flags,
          env,
          sel: liveSel,
          h: helpers
        }) || ({});
        if (hints.pdMode) pdMode = hints.pdMode;
      }
    }
    return {
      flags,
      env,
      pdMode
    };
  };
  const renderCommandLines = (cell, flags, cellEnv, sel, envValues, pdMode = null, mode = "python") => {
    const modelName = resolveModelName(sel);
    let f = [...flags];
    const nnodesFlag = f.find(x => x.split(/[\s=]/)[0] === "--nnodes");
    const nnodesMatch = nnodesFlag && (/^--nnodes(?:\s+|=)(\d+)$/).exec(nnodesFlag.trim());
    const baseNnodes = sel.nodes !== undefined ? parseNnodes(sel.nodes) : cell && cell.nnodes || 1;
    const nnodes = nnodesMatch ? parseInt(nnodesMatch[1], 10) : baseNnodes;
    const multinode = nnodes > 1;
    if (multinode && !f.some(x => x.startsWith("--nnodes"))) {
      const PARALLELISM_ANCHORS = ["--enable-dp-attention", "--dp-size", "--dp", "--tp-size", "--tp"];
      let at = -1;
      for (const anchor of PARALLELISM_ANCHORS) {
        at = f.findIndex(x => x.split(/[\s=]/)[0] === anchor);
        if (at !== -1) break;
      }
      if (at === -1) at = f.findIndex(x => x.startsWith("--model-path"));
      const distPort = pdMode && PD_PORTS[pdMode] ? PD_PORTS[pdMode].dist : 20000;
      f.splice(at + 1, 0, `--nnodes ${nnodes}`, `--node-rank {{NODE_RANK}}`, `--dist-init-addr {{NODE0_IP}}:${distPort}`);
    }
    let cmd;
    if (mode === "docker") {
      const di = config.dockerImages || ({});
      const image = di[`${sel.hw}|${sel.variant}|${sel.quant}`] || di[`${sel.variant}|${sel.quant}`] || di[`${sel.hw}|${sel.quant}|${sel.strategy}`] || di[`${sel.hw}|${sel.quant}`] || di[sel.hw] || "lmsysorg/sglang:dev";
      const dockerRunCommand = typeof config.dockerRunCommand === "function" ? config.dockerRunCommand(sel) : config.dockerRunCommand || "sglang serve";
      const portFlag = f.find(x => x.split(/[\s=]/)[0] === "--port");
      const servePort = portFlag ? portFlag.slice(("--port").length).trim() : "{{PORT}}";
      const hostNetwork = multinode || pdMode || typeof config.dockerHostNetworkWhen === "function" && config.dockerHostNetworkWhen(sel, {
        flags: f,
        env: cellEnv
      });
      const AMD_RDMA_DOCKER_FLAGS = ["--device /dev/infiniband", "--cap-add IPC_LOCK", "--ulimit memlock=-1", "--ulimit stack=67108864", "--ulimit nofile=1048576:1048576"];
      const HW_MULTINODE_DOCKER_FLAGS = {
        "dgx-spark": ["--ulimit memlock=-1:-1", "--cap-add IPC_LOCK", "--device /dev/infiniband"],
        mi300x: AMD_RDMA_DOCKER_FLAGS,
        mi325x: AMD_RDMA_DOCKER_FLAGS,
        mi350x: AMD_RDMA_DOCKER_FLAGS,
        mi355x: AMD_RDMA_DOCKER_FLAGS
      };
      const fabricFlags = HW_MULTINODE_DOCKER_FLAGS[sel.hw] || [];
      const isAmdHw = (/^mi\d/).test(sel.hw || "");
      const HW_NPU_DEVICES = {
        a3: 16,
        a5: 8
      };
      const npuDevices = HW_NPU_DEVICES[sel.hw];
      const davinciLines = devices => {
        const lines = [];
        for (let i = 0; i < devices; i += 4) {
          const group = [];
          for (let k = i; k < Math.min(i + 4, devices); k++) {
            group.push(`--device=/dev/davinci${k}`);
          }
          lines.push("  " + group.join(" "));
        }
        return lines;
      };
      const dockerLines = [...isAmdHw ? ["docker run", "  --device=/dev/kfd --device=/dev/dri", "  --group-add video", "  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined", "  --shm-size 32g"] : npuDevices ? ["docker run --privileged --shm-size=16g", ...davinciLines(npuDevices), "  --device=/dev/davinci_manager", "  --device=/dev/hisi_hdc", "  -v /usr/local/sbin:/usr/local/sbin", "  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver", "  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware", "  -v /etc/ascend_install.info:/etc/ascend_install.info", "  -v /var/queue_schedule:/var/queue_schedule", "  -v ~/.cache/:/root/.cache/"] : ["docker run --gpus all", "  --shm-size 32g"], hostNetwork ? "  --network host" : `  -p ${servePort}:${servePort}`, ...multinode || pdMode ? fabricFlags.map(x => "  " + x) : [], ...npuDevices ? [] : ["  -v ~/.cache/huggingface:/root/.cache/huggingface"], ...(config.dockerMounts || []).map(mount => `  -v ${mount}`), `  --env "HF_TOKEN={{HF_TOKEN}}"`, ...cellEnv.map(e => `  --env ${e}`), "  --ipc=host", `  ${image}`, `  ${dockerRunCommand}`, ...f.map(x => "    " + x)];
      cmd = dockerLines.join(" \\\n");
    } else {
      const flagBlock = f.map(x => "  " + x).join(" \\\n");
      const envBlock = cellEnv.length ? cellEnv.join(" \\\n") + " \\\n" : "";
      cmd = `${envBlock}sglang serve \\\n${flagBlock}`;
    }
    if (multinode && config.multiNodeHints && config.multiNodeHints[sel.hw]) {
      const hint = config.multiNodeHints[sel.hw].map(line => line.length ? "# " + line : "#").join("\n");
      cmd = `${hint}\n${cmd}`;
    }
    cmd = interpolate(cmd, envValues, modelName);
    if (multinode) {
      const header = `# Multi-node (${nnodes} nodes). Run the same command on every node with:\n` + `#   <node-rank> = 0 on the head node, 1..${nnodes - 1} on the others\n` + `#   <node0-ip>  = IP of the head node (reachable from all others)`;
      cmd = `${header}\n${cmd}`;
    }
    if (pdMode === "prefill" || pdMode === "decode") {
      const sibling = pdMode === "prefill" ? "decode" : "prefill";
      const routerCfg = resolveRouter(config.playgroundFeatures && config.playgroundFeatures.pdDisagg, sel);
      const routerPort = routerCfg && routerCfg.port || 8000;
      const routerLine = routerCfg ? `# then front BOTH with the Router shown below.\n` + `# Client traffic (cURL) targets the router (:${routerPort}), not this role server.` : `# then front BOTH with a router; client traffic targets the router, not this role server.`;
      const hicacheCfg = config.playgroundFeatures && config.playgroundFeatures.hicache;
      const pdBackend = findFlagArg(f, "--disaggregation-transfer-backend");
      const hicacheEnabled = f.some(x => x === "--enable-hierarchical-cache");
      const hicacheNotice = hicacheEnabled && hicacheCfg ? (hicacheCfg.notices || []).find(item => {
        if (!item || item.mode !== pdMode) return false;
        if (item.transferBackend && item.transferBackend !== pdBackend) return false;
        return !item.when || matchConstraint(sel, item.when);
      }) : null;
      const noticeLine = hicacheNotice && hicacheNotice.text ? `# Note: ${hicacheNotice.text}\n` : "";
      const banner = `# === PD Disaggregation: ${pdMode.toUpperCase()} role ===\n` + noticeLine + `# Runs the ${pdMode} server. Also run the ${sibling} role on its peer host,\n` + routerLine;
      cmd = `${banner}\n${cmd}`;
    }
    return cmd;
  };
  const computeDiff = (baseStr, pgStr) => {
    const a = baseStr.split("\n");
    const b = pgStr.split("\n");
    const m = a.length, n = b.length;
    const dp = Array(m + 1).fill(null).map(() => new Array(n + 1).fill(0));
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        if (a[i - 1] === b[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1; else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
    const out = [];
    let i = m, j = n;
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
        out.unshift({
          line: a[i - 1],
          kind: "unchanged"
        });
        i--;
        j--;
      } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
        out.unshift({
          line: b[j - 1],
          kind: "added"
        });
        j--;
      } else {
        out.unshift({
          line: a[i - 1],
          kind: "removed"
        });
        i--;
      }
    }
    return out;
  };
  const serializeCell = (sel, env, flags) => {
    const matchEntries = [`hw: ${JSON.stringify(sel.hw)}`, `variant: ${JSON.stringify(sel.variant)}`, `quant: ${JSON.stringify(sel.quant)}`, `strategy: ${JSON.stringify(sel.strategy)}`, `nodes: ${JSON.stringify(sel.nodes)}`].join(", ");
    const fmtList = items => {
      if (!items || items.length === 0) return "[]";
      const lines = items.map(s => `        ${JSON.stringify(s)},`).join("\n");
      return `[\n${lines}\n      ]`;
    };
    return ["    {", `      match: { ${matchEntries} },`, "      verified: true,", `      env: ${fmtList(env)},`, `      flags: ${fmtList(flags)},`, "    },"].join("\n");
  };
  const buildSubmitUrl = (sel, fields) => {
    const gh = config.github || ({});
    const owner = gh.owner || "sgl-project";
    const repo = gh.repo || "sglang";
    const tmpl = gh.issueTemplate || "3-playground-verified-cell.yml";
    const cookbookModel = gh.cookbookModel || "deepseek-ai/deepseek-v4";
    const combo = `${sel.hw} / ${sel.variant} / ${sel.quant} / ${sel.strategy} / ${sel.nodes}`;
    const params = new URLSearchParams({
      template: tmpl,
      title: `[Playground] Verified cell: ${combo}`,
      model: cookbookModel,
      combination: combo,
      "cell-snippet": fields.cellSnippet || "",
      "existing-cell": fields.existingCell || "",
      "sglang-version": fields.sglangVersion || "",
      "bench-result": fields.benchResult || "",
      notes: fields.notes || ""
    });
    return `https://github.com/${owner}/${repo}/issues/new?${params.toString()}`;
  };
  const makeStyles = isDark => ({
    container: {
      maxWidth: "900px",
      margin: "0 auto",
      display: "flex",
      flexDirection: "column",
      gap: "6px"
    },
    card: {
      padding: "6px 10px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#FDBA74" : "#FB923C"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff"
    },
    cardStack: {
      display: "flex",
      flexDirection: "column",
      gap: "6px"
    },
    baseStrip: {
      padding: "8px 12px",
      borderRadius: "4px",
      background: isDark ? "#064e3b" : "#d1fae5",
      color: isDark ? "#a7f3d0" : "#065f46",
      fontSize: "12px",
      display: "flex",
      alignItems: "center",
      gap: "10px"
    },
    title: {
      fontSize: "13px",
      fontWeight: "600",
      color: isDark ? "#e5e7eb" : "inherit",
      marginBottom: "8px"
    },
    compactRow: {
      display: "flex",
      flexWrap: "wrap",
      alignItems: "center",
      gap: "10px",
      rowGap: "4px"
    },
    axisTitle: {
      fontSize: "12px",
      fontWeight: 700,
      color: isDark ? "#FDBA74" : "#C2410C",
      letterSpacing: "0.02em",
      minWidth: "100px",
      flexShrink: 0
    },
    field: {
      display: "inline-flex",
      alignItems: "center",
      gap: "4px"
    },
    fieldLabel: {
      fontSize: "11px",
      fontWeight: 500,
      color: isDark ? "#9ca3af" : "#6b7280"
    },
    select: {
      padding: "2px 6px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "3px",
      fontSize: "12px",
      background: isDark ? "#111827" : "#fff",
      color: isDark ? "#e5e7eb" : "#111827",
      cursor: "pointer",
      lineHeight: "1.4"
    },
    rowFlex: {
      display: "flex",
      flexWrap: "wrap",
      gap: "6px",
      alignItems: "center",
      flex: 1
    },
    subRow: {
      display: "flex",
      alignItems: "center",
      gap: "10px"
    },
    subLabel: {
      fontSize: "11px",
      fontWeight: 600,
      color: isDark ? "#9ca3af" : "#6b7280",
      minWidth: "96px",
      flexShrink: 0,
      letterSpacing: "0.02em"
    },
    chipRow: {
      display: "flex",
      flexWrap: "wrap",
      gap: "6px",
      flex: 1
    },
    chip: {
      padding: "3px 9px",
      border: `1px solid ${isDark ? "#9ca3af" : "#d1d5db"}`,
      borderRadius: "3px",
      cursor: "pointer",
      fontSize: "12px",
      userSelect: "none",
      background: isDark ? "#374151" : "#fff",
      color: isDark ? "#e5e7eb" : "inherit",
      textAlign: "center"
    },
    chipChecked: {
      background: "#D45D44",
      color: "white",
      borderColor: "#D45D44"
    },
    chipDisabled: {
      cursor: "not-allowed",
      opacity: 0.4
    },
    commandWrap: {
      position: "relative",
      background: isDark ? "#111827" : "#f5f5f5",
      borderRadius: "6px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      overflow: "hidden"
    },
    commandHeader: {
      display: "flex",
      flexWrap: "wrap",
      justifyContent: "space-between",
      alignItems: "center",
      gap: "6px 10px",
      padding: "6px 10px",
      borderBottom: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      background: isDark ? "#1f2937" : "#fafafa"
    },
    commandPre: {
      padding: "12px 16px",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
      fontSize: "12px",
      lineHeight: "1.5",
      color: isDark ? "#e5e7eb" : "#374151",
      whiteSpace: "pre-wrap",
      overflowX: "auto",
      margin: 0
    },
    axisNote: {
      margin: "6px 0 0",
      padding: "6px 10px",
      borderRadius: "6px",
      fontSize: "11px",
      lineHeight: "1.45",
      background: isDark ? "#78350f" : "#fef3c7",
      color: isDark ? "#fde68a" : "#92400e",
      border: `1px solid ${isDark ? "#92400e" : "#fcd34d"}`
    },
    mtpWarn: {
      margin: "8px 0 0",
      padding: "8px 12px",
      borderRadius: "8px",
      fontSize: "12px",
      lineHeight: "1.45",
      background: isDark ? "#78350f" : "#fef3c7",
      color: isDark ? "#fde68a" : "#92400e",
      border: `1px solid ${isDark ? "#92400e" : "#fcd34d"}`
    },
    diffLineUnchanged: {
      display: "block"
    },
    diffLineAdded: {
      display: "block",
      background: isDark ? "rgba(16,185,129,0.15)" : "rgba(16,185,129,0.18)",
      color: isDark ? "#a7f3d0" : "#065f46",
      borderLeft: `3px solid #10b981`,
      paddingLeft: "8px",
      marginLeft: "-8px"
    },
    diffLineRemoved: {
      display: "block",
      background: isDark ? "rgba(239,68,68,0.10)" : "rgba(239,68,68,0.10)",
      color: isDark ? "#fca5a5" : "#991b1b",
      textDecoration: "line-through",
      opacity: 0.7,
      borderLeft: `3px solid #ef4444`,
      paddingLeft: "8px",
      marginLeft: "-8px"
    },
    badge: verified => ({
      display: "inline-flex",
      alignItems: "center",
      gap: "6px",
      padding: "2px 8px",
      borderRadius: "10px",
      background: verified ? isDark ? "#064e3b" : "#d1fae5" : isDark ? "#78350f" : "#fef3c7",
      color: verified ? isDark ? "#a7f3d0" : "#065f46" : isDark ? "#fde68a" : "#92400e",
      fontSize: "11px",
      fontWeight: 600
    }),
    badgeDot: verified => ({
      width: "8px",
      height: "8px",
      borderRadius: "50%",
      background: verified ? "#10b981" : "#f59e0b"
    }),
    iconButton: {
      padding: "4px 10px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
      color: isDark ? "#e5e7eb" : "#374151",
      fontSize: "11px",
      fontWeight: 500,
      cursor: "pointer",
      display: "inline-flex",
      alignItems: "center",
      gap: "4px"
    },
    iconRow: {
      display: "inline-flex",
      flexWrap: "wrap",
      gap: "6px"
    },
    runModeWrap: {
      display: "inline-flex",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "10px",
      overflow: "hidden",
      fontSize: "11px",
      fontWeight: 600,
      userSelect: "none"
    },
    runModeChip: active => ({
      padding: "2px 10px",
      cursor: "pointer",
      background: active ? isDark ? "#1f2937" : "#fff" : "transparent",
      color: active ? isDark ? "#e5e7eb" : "#111827" : isDark ? "#9ca3af" : "#6b7280",
      borderRight: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`
    }),
    runModeChipLast: active => ({
      padding: "2px 10px",
      cursor: "pointer",
      background: active ? isDark ? "#1f2937" : "#fff" : "transparent",
      color: active ? isDark ? "#e5e7eb" : "#111827" : isDark ? "#9ca3af" : "#6b7280"
    }),
    headerLeft: {
      display: "inline-flex",
      flexWrap: "wrap",
      alignItems: "center",
      gap: "8px"
    },
    dialog: {
      background: isDark ? "#1f2937" : "#fff",
      color: isDark ? "#e5e7eb" : "#111827",
      borderRadius: "8px",
      padding: "20px",
      maxWidth: "720px",
      width: "92%",
      maxHeight: "calc(100vh - 80px)",
      overflowY: "auto",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      boxShadow: "0 10px 25px rgba(0,0,0,0.25)",
      margin: "auto"
    },
    modalHeader: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      marginBottom: "12px"
    },
    modalTitle: {
      fontSize: "15px",
      fontWeight: 600
    },
    modalCloseBtn: {
      background: "transparent",
      border: "none",
      color: "inherit",
      fontSize: "20px",
      cursor: "pointer",
      padding: "0 6px",
      lineHeight: 1
    },
    formField: {
      display: "flex",
      flexDirection: "column",
      gap: "4px",
      marginBottom: "10px"
    },
    formLabel: {
      fontSize: "12px",
      fontWeight: 500,
      color: isDark ? "#9ca3af" : "#4b5563"
    },
    formInput: {
      padding: "6px 10px",
      fontSize: "13px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "4px",
      background: isDark ? "#111827" : "#fff",
      color: isDark ? "#e5e7eb" : "#111827",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace"
    },
    sectionHeading: {
      fontSize: "12px",
      fontWeight: 600,
      textTransform: "uppercase",
      letterSpacing: "0.04em",
      color: isDark ? "#9ca3af" : "#6b7280",
      margin: "12px 0 6px 0"
    },
    primaryBtn: {
      padding: "6px 14px",
      background: isDark ? "#FDBA74" : "#FB923C",
      color: isDark ? "#7C2D12" : "white",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      fontSize: "13px",
      fontWeight: 500
    },
    resetBtn: {
      marginLeft: "auto",
      padding: "2px 8px",
      fontSize: "11px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "3px",
      background: "transparent",
      color: isDark ? "#9ca3af" : "#6b7280",
      cursor: "pointer"
    },
    switchBaseBtn: {
      padding: "2px 8px",
      fontSize: "11px",
      fontWeight: 600,
      border: `1px solid ${isDark ? "#FDBA74" : "#FB923C"}`,
      borderRadius: "3px",
      background: "transparent",
      color: isDark ? "#FDBA74" : "#C2410C",
      cursor: "pointer"
    },
    matchedHint: {
      fontSize: "11px",
      color: isDark ? "#9ca3af" : "#6b7280",
      marginLeft: "8px",
      display: "inline-flex",
      alignItems: "center",
      gap: "4px"
    },
    matchedSwitchBtn: {
      marginLeft: "4px",
      background: "transparent",
      border: "none",
      padding: 0,
      color: isDark ? "#FDBA74" : "#C2410C",
      cursor: "pointer",
      fontSize: "11px",
      fontWeight: 600,
      textDecoration: "underline",
      textUnderlineOffset: "2px"
    }
  });
  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    const check = () => {
      const html = document.documentElement;
      setIsDark(html.classList.contains("dark") || html.getAttribute("data-theme") === "dark" || html.style.colorScheme === "dark");
    };
    check();
    const observer = new MutationObserver(check);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class", "data-theme", "style"]
    });
    return () => observer.disconnect();
  }, []);
  const [env, setEnv] = useState(() => placeholderDefaults(config.placeholders));
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        setEnv({
          ...placeholderDefaults(config.placeholders),
          ...parsed
        });
      }
    } catch {}
  }, []);
  const saveEnv = next => {
    setEnv(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {}
  };
  const overlayDefaults = () => {
    const out = {};
    for (const d of config.overlayDims || []) {
      const opts = d.options || [];
      out[d.id] = d.default !== undefined ? d.default : opts[0] && opts[0].id || "";
    }
    return out;
  };
  const baseFallback = () => ({
    ...config.cells[0].match,
    ...overlayDefaults()
  });
  const initialBaseFromHash = () => {
    const fallback = baseFallback();
    if (typeof window === "undefined") return {
      ...fallback
    };
    const raw = window.location.hash.replace(/^#/, "");
    if (!raw) return {
      ...fallback
    };
    const params = new URLSearchParams(raw);
    const out = {
      ...fallback
    };
    params.forEach((value, key) => {
      if ((key in out)) out[key] = value;
    });
    return out;
  };
  const [base, setBase] = useState(() => initialBaseFromHash());
  useEffect(() => {
    const onHash = () => setBase(initialBaseFromHash());
    const onSelEvent = e => {
      const fallback = baseFallback();
      const incoming = e && e.detail || ({});
      const next = {
        ...fallback
      };
      for (const k of Object.keys(next)) {
        if (incoming[k] !== undefined) next[k] = incoming[k];
      }
      setBase(next);
    };
    window.addEventListener("hashchange", onHash);
    window.addEventListener("sglang-deploy-sel", onSelEvent);
    return () => {
      window.removeEventListener("hashchange", onHash);
      window.removeEventListener("sglang-deploy-sel", onSelEvent);
    };
  }, []);
  const [pgRatios, setPgRatios] = useState({
    eff: null,
    base: null
  });
  useEffect(() => {
    const onRatio = e => setPgRatios({
      eff: e.detail && e.detail.ratio || null,
      base: e.detail && (e.detail.baseRatio || e.detail.ratio) || null
    });
    window.addEventListener("sglang-k3-mamba-ratio", onRatio);
    return () => window.removeEventListener("sglang-k3-mamba-ratio", onRatio);
  }, []);
  const initialDeltas = () => {
    const out = {};
    for (const [axisId, handler] of Object.entries(AXIS_HANDLERS)) {
      const fc = pgFeatures[axisId];
      if (fc) out[axisId] = handler.initState(fc, base);
    }
    return out;
  };
  const [deltas, setDeltas] = useState(initialDeltas);
  useEffect(() => {
    setDeltas(initialDeltas());
  }, [Object.keys(base).sort().map(k => `${k}=${base[k]}`).join("&")]);
  const [modal, setModal] = useState(null);
  const openDialog = el => {
    if (el && !el.open) {
      try {
        el.showModal();
      } catch {}
    }
  };
  const onDialogClick = e => {
    if (e.target !== e.currentTarget) return;
    const r = e.currentTarget.getBoundingClientRect();
    const {clientX: x, clientY: y} = e;
    if (x < r.left || x > r.right || y < r.top || y > r.bottom) setModal(null);
  };
  useEffect(() => {
    const ID = "__playground_dialog_backdrop";
    if (document.getElementById(ID)) return undefined;
    const style = document.createElement("style");
    style.id = ID;
    style.textContent = `dialog::backdrop { background: rgba(0, 0, 0, 0.5); }`;
    document.head.appendChild(style);
    return () => {
      const el = document.getElementById(ID);
      if (el) el.remove();
    };
  }, []);
  const [copied, setCopied] = useState(false);
  const [curlCopied, setCurlCopied] = useState(false);
  const [routerCopied, setRouterCopied] = useState(false);
  const [envDraft, setEnvDraft] = useState(env);
  useEffect(() => {
    if (modal === "env") setEnvDraft(env);
  }, [modal, env]);
  const [runMode, setRunMode] = useState("python");
  const [submitDraft, setSubmitDraft] = useState({
    sglangVersion: "",
    benchResult: "",
    notes: ""
  });
  const [submitAttest, setSubmitAttest] = useState({
    ranCommand: false,
    reachedReady: false,
    outputCorrect: false
  });
  useEffect(() => {
    if (modal === "submit") {
      setSubmitDraft({
        sglangVersion: "",
        benchResult: "",
        notes: ""
      });
      setSubmitAttest({
        ranCommand: false,
        reachedReady: false,
        outputCorrect: false
      });
    }
  }, [modal]);
  const s = makeStyles(isDark);
  const baseCell = withOverlay(findCell(config.cells, base), base);
  const modelName = resolveModelName(base);
  const derivedMap = {};
  if (baseCell) {
    for (const [axisId, handler] of Object.entries(AXIS_HANDLERS)) {
      const fc = pgFeatures[axisId];
      if (!fc || !handler.deriveFromBase) continue;
      derivedMap[axisId] = handler.deriveFromBase(baseCell, fc, helpers);
    }
  }
  const attnDelta = deltas.attention || ({});
  const attnDerived = derivedMap.attention || ({});
  const attnKnobs = (pgFeatures.attention || ({})).knobs || [];
  const effTp = attnDelta.tp !== null && attnDelta.tp !== undefined ? attnDelta.tp : attnDerived.tp !== undefined ? attnDerived.tp : null;
  const staleExplicit = (knobId, picked) => {
    const knob = attnKnobs.find(k => k.id === knobId);
    const e = knob ? findEntry(knob.values || [], picked) : null;
    return !!(e !== null && e !== undefined && evaluateChip(e, {
      ...base,
      effTp
    }).disabled);
  };
  const effDpAttn = attnDelta.dpAttn !== null && attnDelta.dpAttn !== undefined ? staleExplicit("dpAttn", attnDelta.dpAttn) ? null : attnDelta.dpAttn : attnDerived.dpAttn !== undefined ? attnDerived.dpAttn : null;
  const dpAttnOn = effDpAttn === true || typeof effDpAttn === "number" && effDpAttn > 0;
  const dpDegEff = typeof effDpAttn === "number" && effDpAttn > 0 ? effDpAttn : 1;
  const cpKnobFreeSize = !!(attnKnobs.find(k => k.id === "cp") || ({})).freeSize;
  const cpSizeTarget = !cpKnobFreeSize && dpDegEff === 1 && typeof effTp === "number" && effTp > 0 ? effTp : null;
  const cpSizeStale = v => typeof v === "number" && v > 1 && cpSizeTarget !== null && v !== cpSizeTarget;
  const effCp = attnDelta.cp !== null && attnDelta.cp !== undefined ? staleExplicit("cp", attnDelta.cp) || cpSizeStale(attnDelta.cp) ? null : attnDelta.cp : attnDerived.cp !== undefined ? attnDerived.cp : null;
  const cpOn = typeof effCp === "number" && effCp > 1;
  const cpStrategy = (attnDelta.cpStrategy && !staleExplicit("cpStrategy", attnDelta.cpStrategy) ? attnDelta.cpStrategy : attnDerived.cpStrategy !== undefined ? attnDerived.cpStrategy : null) || "interleave";
  const constraintEffective = baseCell ? applyAllDeltas(baseCell.flags, baseCell.env, deltas, base, derivedMap) : null;
  const pdCardOwnsMode = (pgFeatures.pdDisagg && pgFeatures.pdDisagg.modes || []).length > 0;
  const pdMode = pdCardOwnsMode ? constraintEffective && constraintEffective.pdMode || "off" : base.pdMode || "off";
  const specAlgorithm = constraintEffective ? (findFlagArg(constraintEffective.flags, "--speculative-algorithm") || "").toUpperCase() || null : null;
  const constraintBase = {
    ...base,
    dpAttnOn,
    cpOn,
    cpStrategy,
    cpSizeTarget,
    effTp,
    pdMode,
    specAlgorithm
  };
  let baseCommand = "";
  let playgroundCommand = "";
  let diffLines = [];
  let pgFlagsLatest = [];
  let pgEnvLatest = [];
  const withRatio = (fl, value) => {
    if (!value) return fl;
    if (fl.some(f => f.startsWith("--mamba-full-memory-ratio") || f.startsWith("--max-mamba-cache-size"))) return fl;
    const out = [...fl];
    const line = `--mamba-full-memory-ratio ${value}`;
    const i = out.findIndex(f => f.startsWith("--host"));
    if (i >= 0) out.splice(i, 0, line); else out.push(line);
    return out;
  };
  if (baseCell) {
    baseCommand = renderCommandLines(baseCell, withRatio(baseCell.flags, pgRatios.base), baseCell.env, base, env, null, runMode);
    const {flags: pgFlags, env: pgEnv, pdMode} = applyAllDeltas(baseCell.flags, baseCell.env, deltas, base, derivedMap);
    pgFlagsLatest = pgFlags;
    pgEnvLatest = pgEnv;
    playgroundCommand = renderCommandLines(baseCell, withRatio(pgFlags, pgRatios.eff), pgEnv, base, env, pdMode, runMode);
    diffLines = computeDiff(baseCommand, playgroundCommand);
  }
  const effectiveKey = pgFlagsLatest.join("\n") + " " + pgEnvLatest.join("\n") + " " + (baseCell ? baseCell.flags.join("\n") + " " + baseCell.env.join("\n") : "");
  useEffect(() => {
    if (typeof window === "undefined" || !baseCell) return;
    window.dispatchEvent(new CustomEvent("sglang-k3-effective-config", {
      detail: {
        flags: pgFlagsLatest,
        env: pgEnvLatest,
        baseFlags: baseCell.flags,
        baseEnv: baseCell.env
      }
    }));
  }, [effectiveKey]);
  const matchedCell = baseCell ? findMatchingCell(config.cells, base, pgEnvLatest, pgFlagsLatest) : null;
  const playgroundVerified = !!(matchedCell && matchedCell.verified);
  const matchedSiblingCell = matchedCell && DIMENSIONS.some(d => matchedCell.match[d] !== base[d]) ? matchedCell : null;
  const pgSpecAlgoFlag = pgFlagsLatest.find(f => f.split(/[\s=]/)[0] === "--speculative-algorithm");
  const pgPdRole = pdMode === "prefill" || pdMode === "decode";
  const pgSpecHint = !!pgSpecAlgoFlag && (pgPdRole || !pgFlagsLatest.some(f => f.split(/[\s=]/)[0] === "--max-running-requests"));
  const specAlgoLabels = {
    EAGLE: "MTP",
    EAGLE3: "MTP",
    FROZEN_KV_MTP: "MTP",
    DSPARK: "DSpark",
    DFLASH: "DFlash",
    NGRAM: "N-gram",
    STANDALONE: "standalone draft"
  };
  const pgSpecAlgoValue = pgSpecAlgoFlag ? pgSpecAlgoFlag.split(/[\s=]/).filter(Boolean)[1] || "" : "";
  const pgSpecAlgoName = specAlgoLabels[pgSpecAlgoValue.toUpperCase()] || pgSpecAlgoValue || "MTP";
  const pgCpDpHint = cpEnabledIn(pgFlagsLatest) && (bakedCpStrategy(pgFlagsLatest) || "interleave") === "interleave" && pgFlagsLatest.some(f => f.split(/[\s=]/)[0] === "--enable-dp-attention");
  const proposedCellSnippet = baseCell ? serializeCell(base, pgEnvLatest, pgFlagsLatest) : "";
  const existingCellSnippet = baseCell ? serializeCell(base, baseCell.env || [], baseCell.flags) : "";
  const submitUrl = baseCell ? buildSubmitUrl(base, {
    cellSnippet: proposedCellSnippet,
    existingCell: existingCellSnippet,
    sglangVersion: submitDraft.sglangVersion,
    benchResult: submitDraft.benchResult,
    notes: submitDraft.notes
  }) : "";
  const submitReady = submitAttest.ranCommand && submitAttest.reachedReady && submitAttest.outputCorrect && submitDraft.sglangVersion.trim().length > 0;
  const pdRouter = pdMode !== "off" && resolveRouter(config.playgroundFeatures && config.playgroundFeatures.pdDisagg, base) || null;
  const curlEnv = pdRouter && pdRouter.port != null ? {
    ...env,
    CURL_PORT: String(pdRouter.port)
  } : env;
  const curlText = interpolate(config.curl || "", curlEnv, modelName);
  const routerText = pdRouter && pdRouter.command ? interpolate(pdRouter.command, {
    ...env,
    PREFILL_PORT: PD_PORTS.prefill.serve,
    DECODE_PORT: PD_PORTS.decode.serve,
    ROUTER_PORT: pdRouter.port
  }, modelName) : "";
  const resetAll = () => setDeltas(initialDeltas());
  const placeholderGroups = (() => {
    const out = {
      command: [],
      curl: []
    };
    for (const [key, meta] of Object.entries(config.placeholders || ({}))) {
      (out[meta.target] || (out[meta.target] = [])).push({
        key,
        ...meta
      });
    }
    return out;
  })();
  const handleCopy = () => {
    navigator.clipboard.writeText(playgroundCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };
  const copyCurl = () => {
    navigator.clipboard.writeText(curlText);
    setCurlCopied(true);
    setTimeout(() => setCurlCopied(false), 1200);
  };
  const baseSummary = baseCell ? Object.entries(base).filter(([, v]) => v !== undefined && v !== "").map(([k, v]) => k === "hw" ? String(v).toUpperCase() : String(v)).join(" · ") : "(no verified cell at the current Deploy selection — showing playground only)";
  const renderChip = (label, current, value, onPick, opts = {}) => {
    const checked = current === value;
    const disabled = !!opts.disabled;
    return <span key={`${label}-${value === null ? "auto" : value}`} style={{
      ...s.chip,
      ...checked ? s.chipChecked : {},
      ...disabled ? s.chipDisabled : {}
    }} title={disabled ? opts.disabledReason || "Not available" : ""} onClick={() => {
      if (!disabled) onPick(value);
    }}>
        {label}
      </span>;
  };
  const renderSelect = (current, entries, onPick, base, labelFor, opts = {}) => {
    const hideSet = new Set(opts.hideValues || []);
    const items = [];
    for (const entry of entries || []) {
      const c = helpers.evaluateChip(entry, base);
      if (c.hidden) continue;
      if (hideSet.has(c.value)) continue;
      const lbl = labelFor ? labelFor(c) : c.label !== undefined ? c.label : c.value === null ? "Auto" : String(c.value);
      items.push({
        ...c,
        label: lbl
      });
    }
    let idx = items.findIndex(c => c.value === current);
    if (idx === -1) idx = 0;
    return <select style={{
      ...s.select,
      ...opts.disabled ? s.chipDisabled : {}
    }} disabled={!!opts.disabled} title={opts.disabled ? opts.disabledReason || "Not available" : ""} value={idx} onChange={e => {
      const next = items[parseInt(e.target.value, 10)];
      if (next && !next.disabled) onPick(next.value);
    }}>
        {items.map((c, i) => <option key={i} value={i} disabled={c.disabled}>
            {c.label}{c.disabled ? " (n/a)" : ""}
          </option>)}
      </select>;
  };
  return <div style={s.container} className="not-prose">
      {}
      <div style={s.baseStrip}>
        <span style={{
    fontWeight: 600
  }}>Inherited base from Deployment:</span>
        <code style={{
    fontFamily: "Menlo, monospace"
  }}>{baseSummary}</code>
        {}
        <button type="button" style={s.switchBaseBtn} onClick={() => {
    const el = document.getElementById(DEPLOYMENT_COMPONENT_ID) || document.getElementById("deployment") || document.getElementById("deploy");
    if (el) el.scrollIntoView({
      behavior: "smooth",
      block: "start"
    });
  }}>
          ↑ Switch base
        </button>
        <button style={s.resetBtn} onClick={resetAll}>Reset all overrides</button>
      </div>

      {}
      {Object.entries(AXIS_HANDLERS).map(([axisId, handler]) => {
    const fc = pgFeatures[axisId];
    if (!fc) return null;
    if (typeof fc.showWhen === "function" && !fc.showWhen(constraintBase)) return null;
    const setValue = next => setDeltas(d => ({
      ...d,
      [axisId]: next
    }));
    return handler.render({
      axisId,
      value: deltas[axisId],
      setValue,
      fc,
      base: constraintBase,
      s,
      h: helpers,
      renderChip,
      renderSelect,
      derived: derivedMap[axisId] || null
    });
  })}

      {}
      <div style={s.card}>
        <div style={s.title}>Playground Command (compare with base)</div>
        <div style={s.commandWrap}>
          <div style={s.commandHeader}>
            <div style={s.headerLeft}>
              <div style={s.badge(playgroundVerified)}>
                <span style={s.badgeDot(playgroundVerified)} />
                {playgroundVerified ? "Verified" : "Not Verified"}
              </div>
              {}
              {matchedSiblingCell && <span style={s.matchedHint}>
                  matches <code style={{
    fontFamily: "Menlo, monospace"
  }}>
                    {matchedSiblingCell.match.strategy}
                  </code>
                  <button type="button" style={s.matchedSwitchBtn} onClick={() => {
    const m = matchedSiblingCell.match;
    setDeltas(initialDeltas());
    const hash = new URLSearchParams(m).toString();
    window.location.hash = hash;
    window.dispatchEvent(new CustomEvent("sglang-deploy-sel", {
      detail: m
    }));
  }}>
                    switch base →
                  </button>
                </span>}
              <div style={s.runModeWrap} role="tablist" aria-label="Output format">
                <span style={s.runModeChip(runMode === "python")} onClick={() => setRunMode("python")} role="tab" aria-selected={runMode === "python"}>
                  Python
                </span>
                <span style={s.runModeChipLast(runMode === "docker")} onClick={() => setRunMode("docker")} role="tab" aria-selected={runMode === "docker"}>
                  Docker
                </span>
              </div>
            </div>
            <div style={s.iconRow}>
              <button style={s.iconButton} onClick={handleCopy}>
                {copied ? "✓ Copied" : "⧉ Copy"}
              </button>
              <button style={s.iconButton} onClick={() => setModal("curl")}>$ cURL</button>
              <button style={s.iconButton} onClick={() => setModal("env")}>⚙ Env</button>
              {}
              {!playgroundVerified && baseCell && <button style={{
    ...s.iconButton,
    borderColor: isDark ? "#FDBA74" : "#FB923C",
    color: isDark ? "#FDBA74" : "#C2410C",
    fontWeight: 600
  }} onClick={() => setModal("submit")} title="I verified this command on my hardware — open a pre-filled GitHub issue to land it as a cookbook cell.">
                  Submit ↗
                </button>}
            </div>
          </div>
          <pre style={s.commandPre}>
            {baseCell ? diffLines.map((d, i) => <span key={i} style={d.kind === "added" ? s.diffLineAdded : d.kind === "removed" ? s.diffLineRemoved : s.diffLineUnchanged}>
                {d.kind === "added" ? "+ " : d.kind === "removed" ? "- " : "  "}
                {d.line}{"\n"}
              </span>) : "# No verified base cell at the current Deployment selection.\n# Pick a supported hardware/variant in the Deployment panel to populate the playground base."}
          </pre>
          {pgSpecHint && <div style={s.mtpWarn}>
              {pgPdRole ? <>⚠️ Speculative decoding ({pgSpecAlgoName}) is on — for a target concurrency of N, set <code>--max-running-requests &lt;N*2&gt;</code> on <strong>both</strong> the prefill and decode roles. That ceiling is server-wide and floor-divided by <code>attn_dp_size</code>, so size the decode graphs to the per-rank batch it leaves: <code>--cuda-graph-bs-decode</code> up to <code>N*2 / dp_size</code>.</> : <>⚠️ Speculative decoding ({pgSpecAlgoName}) is on — SGLang resets <code>--max-running-requests</code> to <strong>48</strong> when it isn't set. Add <code>--max-running-requests &lt;N&gt;</code> sized for your target concurrency.</>}
            </div>}
          {pgCpDpHint && <div style={s.mtpWarn}>
              ⚠️ Interleave prefill-CP together with DP-Attention: current SGLang releases assert <code>dp_size == 1</code> for the interleave layout, so this command fails at startup. Combined CP + DP-Attention support is planned upstream — keep one of the two off until it lands.
            </div>}
        </div>
      </div>

      {}
      {pdRouter && routerText && <div style={s.card}>
          <div style={s.title}>Router</div>
          <div style={{
    fontSize: 11,
    opacity: 0.7,
    margin: "0 0 6px"
  }}>
            Run after both roles are up. Substitute <code>{"<prefill-host>"}</code> /{" "}
            <code>{"<decode-host>"}</code> with reachable hosts (both <code>127.0.0.1</code>{" "}
            on a same-host deployment). Client traffic (cURL) targets this router.
          </div>
          <div style={s.commandWrap}>
            <div style={s.commandHeader}>
              <div style={{
    fontSize: 11,
    opacity: 0.7
  }}>port {pdRouter.port}</div>
              <button style={s.iconButton} onClick={() => {
    navigator.clipboard.writeText(routerText);
    setRouterCopied(true);
    setTimeout(() => setRouterCopied(false), 1200);
  }}>
                {routerCopied ? "✓ Copied" : "⧉ Copy"}
              </button>
            </div>
            <pre style={s.commandPre}>{routerText}</pre>
          </div>
        </div>}

      {}
      {modal === "curl" && <dialog ref={openDialog} style={s.dialog} onClose={() => setModal(null)} onClick={onDialogClick}>
          <div style={s.modalHeader}>
            <div style={s.modalTitle}>cURL example</div>
            <button style={s.modalCloseBtn} onClick={() => setModal(null)} aria-label="Close">×</button>
          </div>
          <div style={s.commandWrap}>
            <div style={s.commandHeader}>
              <div style={{
    fontSize: 11,
    opacity: 0.7
  }}>
                Model: <code>{modelName || "(unresolved)"}</code>
              </div>
              <button style={s.iconButton} onClick={copyCurl}>
                {curlCopied ? "✓ Copied" : "⧉ Copy"}
              </button>
            </div>
            <pre style={s.commandPre}>{curlText}</pre>
          </div>
          {pdRouter && <p style={{
    fontSize: 11,
    opacity: 0.85,
    marginTop: 8
  }}>
              <strong>PD-Disaggregation active</strong> — this targets the router on
              {" "}<code>:{pdRouter.port}</code>; client traffic must not hit the role
              servers directly.
            </p>}
          <p style={{
    fontSize: 11,
    opacity: 0.7,
    marginTop: 8
  }}>
            Edit <code>CURL_HOST</code> / <code>CURL_PORT</code> in the Env panel.
          </p>
        </dialog>}

      {}
      {modal === "env" && <dialog ref={openDialog} style={s.dialog} onClose={() => setModal(null)} onClick={onDialogClick}>
          <div style={s.modalHeader}>
            <div style={s.modalTitle}>Env / placeholder values</div>
            <button style={s.modalCloseBtn} onClick={() => setModal(null)} aria-label="Close">×</button>
          </div>
            {placeholderGroups.curl.length > 0 && <div>
                <div style={s.sectionHeading}>cURL placeholders</div>
                {placeholderGroups.curl.map(({key, label}) => <div key={key} style={s.formField}>
                    <label style={s.formLabel}>
                      {label} <code style={{
    opacity: 0.6
  }}>{`{{${key}}}`}</code>
                    </label>
                    <input style={s.formInput} value={envDraft[key] ?? ""} onChange={e => setEnvDraft({
    ...envDraft,
    [key]: e.target.value
  })} />
                  </div>)}
              </div>}
            {placeholderGroups.command.length > 0 && <div>
                <div style={s.sectionHeading}>Command placeholders</div>
                {placeholderGroups.command.map(({key, label}) => <div key={key} style={s.formField}>
                    <label style={s.formLabel}>
                      {label} <code style={{
    opacity: 0.6
  }}>{`{{${key}}}`}</code>
                    </label>
                    <input style={s.formInput} value={envDraft[key] ?? ""} onChange={e => setEnvDraft({
    ...envDraft,
    [key]: e.target.value
  })} />
                  </div>)}
              </div>}
            <div style={{
    display: "flex",
    justifyContent: "flex-end",
    gap: 8,
    marginTop: 16
  }}>
              <button style={{
    ...s.iconButton,
    padding: "6px 14px"
  }} onClick={() => setModal(null)}>Cancel</button>
              <button style={s.primaryBtn} onClick={() => {
    saveEnv(envDraft);
    setModal(null);
  }}>Save</button>
            </div>
          <p style={{
    fontSize: 11,
    opacity: 0.7,
    marginTop: 10
  }}>
            Values persist in localStorage and are shared with the Deployment panel.
          </p>
        </dialog>}

      {}
      {modal === "submit" && <dialog ref={openDialog} style={s.dialog} onClose={() => setModal(null)} onClick={onDialogClick}>
            <div style={s.modalHeader}>
              <div style={s.modalTitle}>Submit verified cell</div>
              <button style={s.modalCloseBtn} onClick={() => setModal(null)} aria-label="Close">×</button>
            </div>
            <p style={{
    fontSize: 12,
    opacity: 0.85,
    marginTop: 0,
    marginBottom: 12
  }}>
              You've put together a combination that isn't in the verified
              catalog yet. After you've run the command end-to-end on the
              target hardware, this submits a pre-filled GitHub Issue that a
              maintainer can convert into a PR.
            </p>

            <div style={s.sectionHeading}>Combination</div>
            <code style={{
    fontFamily: "Menlo, monospace",
    fontSize: 12
  }}>
              {base.hw} / {base.variant} / {base.quant} / {base.strategy} / {base.nodes}
            </code>
            {}
            {(() => {
    const adds = diffLines.filter(d => d.kind === "added");
    const rems = diffLines.filter(d => d.kind === "removed");
    if (adds.length === 0 && rems.length === 0) return null;
    return <>
                  <div style={{
      ...s.sectionHeading,
      marginTop: 10
    }}>
                    Overrides vs base ({adds.length} added · {rems.length} removed)
                  </div>
                  <pre style={{
      margin: 0,
      padding: "8px 10px",
      background: isDark ? "#111827" : "#f5f5f5",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderRadius: 4,
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
      fontSize: 12,
      lineHeight: 1.4,
      maxHeight: 160,
      overflowY: "auto",
      whiteSpace: "pre-wrap"
    }}>
                    {[...rems, ...adds].map((d, i) => <div key={i} style={d.kind === "added" ? s.diffLineAdded : s.diffLineRemoved}>
                        {d.kind === "added" ? "+ " : "- "}
                        {d.line.replace(/^\s*/, "")}
                      </div>)}
                  </pre>
                </>;
  })()}

            <div style={{
    ...s.sectionHeading,
    marginTop: 14
  }}>Attestation (all required)</div>
            <div style={s.formField}>
              <label style={{
    fontSize: 12,
    display: "flex",
    alignItems: "flex-start",
    gap: 6
  }}>
                <input type="checkbox" checked={submitAttest.ranCommand} onChange={e => setSubmitAttest({
    ...submitAttest,
    ranCommand: e.target.checked
  })} />
                I ran this exact command on the listed hardware.
              </label>
              <label style={{
    fontSize: 12,
    display: "flex",
    alignItems: "flex-start",
    gap: 6
  }}>
                <input type="checkbox" checked={submitAttest.reachedReady} onChange={e => setSubmitAttest({
    ...submitAttest,
    reachedReady: e.target.checked
  })} />
                The server reached READY and answered a cURL request successfully.
              </label>
              <label style={{
    fontSize: 12,
    display: "flex",
    alignItems: "flex-start",
    gap: 6
  }}>
                <input type="checkbox" checked={submitAttest.outputCorrect} onChange={e => setSubmitAttest({
    ...submitAttest,
    outputCorrect: e.target.checked
  })} />
                Output looked correct on at least one prompt.
              </label>
            </div>

            <div style={{
    ...s.sectionHeading,
    marginTop: 14
  }}>SGLang version (required)</div>
            <input style={{
    ...s.formInput,
    width: "100%",
    boxSizing: "border-box"
  }} placeholder="sglang==0.5.4  (or git SHA abc1234)" value={submitDraft.sglangVersion} onChange={e => setSubmitDraft({
    ...submitDraft,
    sglangVersion: e.target.value
  })} />

            <div style={{
    ...s.sectionHeading,
    marginTop: 14
  }}>Benchmark result (optional)</div>
            <input style={{
    ...s.formInput,
    width: "100%",
    boxSizing: "border-box"
  }} placeholder="TTFT 95 ms / TPOT 18 ms / 1820 tok/s @ bs=64" value={submitDraft.benchResult} onChange={e => setSubmitDraft({
    ...submitDraft,
    benchResult: e.target.value
  })} />

            <div style={{
    ...s.sectionHeading,
    marginTop: 14
  }}>Notes / caveats (optional)</div>
            <textarea style={{
    ...s.formInput,
    width: "100%",
    boxSizing: "border-box",
    minHeight: 110,
    resize: "vertical",
    fontFamily: "inherit"
  }} placeholder="Cluster config, env-var quirks, NIC mappings, multi-node bootstrap details, …" value={submitDraft.notes} onChange={e => setSubmitDraft({
    ...submitDraft,
    notes: e.target.value
  })} />

            <div style={{
    display: "flex",
    justifyContent: "flex-end",
    gap: 8,
    marginTop: 16,
    alignItems: "center"
  }}>
              {!submitReady && <span style={{
    fontSize: 11,
    opacity: 0.7,
    marginRight: "auto"
  }}>
                  Tick all attestations and fill SGLang version to enable submit.
                </span>}
              <button style={{
    ...s.iconButton,
    padding: "6px 14px"
  }} onClick={() => setModal(null)}>Cancel</button>
              <a href={submitReady ? submitUrl : undefined} target="_blank" rel="noopener noreferrer" onClick={e => {
    if (!submitReady) e.preventDefault(); else setModal(null);
  }} style={{
    ...s.primaryBtn,
    textDecoration: "none",
    display: "inline-flex",
    alignItems: "center",
    opacity: submitReady ? 1 : 0.4,
    cursor: submitReady ? "pointer" : "not-allowed"
  }}>
                Open submission on GitHub →
              </a>
            </div>
          <p style={{
    fontSize: 11,
    opacity: 0.7,
    marginTop: 10
  }}>
            The CTA opens a pre-filled GitHub Issue using the
            <code> 3-playground-verified-cell.yml</code> template. A
            maintainer with the listed hardware will review and convert it
            into a cookbook PR.
          </p>
        </dialog>}
    </div>;
};

export const Qwen38MambaRatioCalculator = () => {
  const [isDark, setIsDark] = useState(false);
  const [requestLength, setRequestLength] = useState("5120");
  const [targetConcurrency, setTargetConcurrency] = useState("64");
  const [copied, setCopied] = useState(false);
  const [cfg, setCfg] = useState({
    flags: [],
    env: [],
    baseFlags: [],
    baseEnv: []
  });
  useEffect(() => {
    const checkTheme = () => {
      const html = document.documentElement;
      setIsDark(html.classList.contains("dark") || html.getAttribute("data-theme") === "dark" || html.style.colorScheme === "dark");
    };
    checkTheme();
    const observer = new MutationObserver(checkTheme);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class", "data-theme", "style"]
    });
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const onCfg = e => setCfg({
      flags: e.detail && e.detail.flags || [],
      env: e.detail && e.detail.env || [],
      baseFlags: e.detail && e.detail.baseFlags || [],
      baseEnv: e.detail && e.detail.baseEnv || []
    });
    window.addEventListener("sglang-k3-effective-config", onCfg);
    return () => window.removeEventListener("sglang-k3-effective-config", onCfg);
  }, []);
  const [quant, setQuant] = useState("nvfp4-bf16-head");
  useEffect(() => {
    const onSel = e => {
      if (e.detail && e.detail.quant) setQuant(e.detail.quant);
    };
    window.addEventListener("sglang-deploy-sel", onSel);
    return () => window.removeEventListener("sglang-deploy-sel", onSel);
  }, []);
  const L = Number.parseFloat(requestLength);
  const C = Number.parseFloat(targetConcurrency);
  const derive = (flags, env) => {
    const flagArg = name => {
      for (const f of flags) {
        const parts = f.split(/\s+/);
        if (parts[0] === name) return parts[1];
      }
      return null;
    };
    const hasFlag = name => flags.some(f => f.split(/[\s=]/)[0] === name);
    const tp = Number(flagArg("--tp")) || Number(flagArg("--tp-size")) || 1;
    const kvFlag = flagArg("--kv-cache-dtype");
    const kvDtype = kvFlag === "fp8_e4m3" ? "fp8_e4m3" : kvFlag === "bfloat16" || kvFlag === "bf16" ? "bfloat16" : quant === "nvfp4-bf16-head" || quant === "nvfp4-fp4-head" ? "fp8_e4m3" : "bfloat16";
    const ssmFlag = flagArg("--mamba-ssm-dtype");
    const ssmDtype = ssmFlag === "bfloat16" || ssmFlag === "float16" ? ssmFlag : "float32";
    const radixOff = hasFlag("--disable-radix-cache");
    const strategyFlag = flagArg("--mamba-radix-cache-strategy");
    const strategy = strategyFlag === "no_buffer" || strategyFlag === "extra_buffer_lazy" ? strategyFlag : "extra_buffer";
    const skipLock = env.some(e => e.startsWith("SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"));
    const overlapOff = hasFlag("--disable-overlap-schedule") || (Number(flagArg("--pp-size")) || 1) > 1;
    const slots = radixOff ? 1 : strategy === "no_buffer" ? 3 : 3 - (skipLock ? 1 : 0) + (overlapOff || strategy === "extra_buffer_lazy" ? 1 : 2);
    const specOn = hasFlag("--speculative-algorithm");
    const algo = (flagArg("--speculative-algorithm") || "").toUpperCase();
    const replaySpec = hasFlag("--enable-linear-replayssm-spec");
    const dsparkBlock = Number(flagArg("--speculative-dspark-block-size")) || 7;
    const drafts = !specOn || replaySpec ? 0 : algo === "DSPARK" ? dsparkBlock + 1 : Number(flagArg("--speculative-num-draft-tokens")) || 4;
    const ssmBytes = ssmDtype === "float32" ? 4 : 2;
    const kvBytes = kvDtype === "fp8_e4m3" ? 1 : 2;
    const stateBytesPerSlot = 48 * (48 * 128 * 128 * ssmBytes + 10240 * 3 * 2);
    const kvBytesPerToken = 16 * 4 * 256 * 2 * kvBytes;
    const ratio = (slots + drafts) * stateBytesPerSlot / (kvBytesPerToken * L);
    return {
      ratio,
      tp,
      kvDtype,
      ssmDtype,
      radixOff,
      strategy,
      slots,
      specOn,
      drafts,
      stateBytesPerSlot,
      kvBytesPerToken
    };
  };
  const eff = derive(cfg.flags, cfg.env);
  const bs = derive(cfg.baseFlags.length ? cfg.baseFlags : cfg.flags, cfg.baseFlags.length ? cfg.baseEnv : cfg.env);
  const {ratio, tp, kvDtype, ssmDtype, radixOff, strategy, slots, specOn, drafts, stateBytesPerSlot, kvBytesPerToken} = eff;
  const valid = Number.isFinite(ratio) && ratio > 0 && L > 0 && tp === 1;
  const baseValid = Number.isFinite(bs.ratio) && bs.ratio > 0 && L > 0 && bs.tp === 1;
  const pin = Math.ceil(C * slots);
  const pinValid = valid && Number.isFinite(pin) && pin > 0 && C > 0;
  const formatRatio = value => (Math.round(value * 100) / 100).toString();
  const ratioStr = valid ? formatRatio(ratio) : "—";
  const baseRatioStr = baseValid ? formatRatio(bs.ratio) : "—";
  const flagText = valid ? pinValid ? `--mamba-full-memory-ratio ${ratioStr}   # or: --max-mamba-cache-size ${pin}` : `--mamba-full-memory-ratio ${ratioStr}` : "";
  useEffect(() => {
    window.dispatchEvent(new CustomEvent("sglang-k3-mamba-ratio", {
      detail: {
        ratio: valid ? ratioStr : null,
        baseRatio: baseValid ? baseRatioStr : null
      }
    }));
  }, [ratioStr, valid, baseRatioStr, baseValid]);
  const copy = () => {
    if (!valid) return;
    navigator.clipboard.writeText(flagText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    });
  };
  const colors = {
    border: isDark ? "#374151" : "#e5e7eb",
    panel: isDark ? "#1f2937" : "#ffffff",
    input: isDark ? "#111827" : "#f8fafc",
    text: isDark ? "#e5e7eb" : "#1f2937",
    muted: isDark ? "#9ca3af" : "#64748b",
    accent: isDark ? "#E85D4D" : "#D45D44",
    error: isDark ? "#fca5a5" : "#b91c1c"
  };
  const inputStyle = {
    width: "100%",
    boxSizing: "border-box",
    padding: "8px 10px",
    border: `1px solid ${colors.border}`,
    borderRadius: "5px",
    background: colors.input,
    color: colors.text,
    fontSize: "13px"
  };
  const labelStyle = {
    display: "flex",
    flexDirection: "column",
    gap: "5px",
    fontSize: "12px",
    fontWeight: 600
  };
  const chipStyle = {
    padding: "3px 9px",
    border: `1px solid ${colors.border}`,
    borderRadius: "999px",
    background: colors.input,
    color: colors.text,
    fontSize: "12px",
    whiteSpace: "nowrap"
  };
  const derivedChips = [`KV ${kvDtype === "fp8_e4m3" ? "FP8" : "BF16"}`, `State ${ssmDtype === "float32" ? "FP32" : ssmDtype === "bfloat16" ? "BF16" : "FP16"}`, radixOff ? "Radix off (S = 1)" : `${strategy} (S = ${slots})`, specOn ? `Spec on (D = ${drafts})` : "NOSPEC"];
  return <div className="not-prose" style={{
    display: "grid",
    gap: "12px",
    padding: "14px",
    border: `1px solid ${colors.border}`,
    borderRadius: "8px",
    background: colors.panel,
    color: colors.text
  }}>
      <div style={{
    display: "grid",
    gridTemplateColumns: "minmax(180px, 240px) minmax(150px, 200px) 1fr",
    gap: "14px",
    alignItems: "start"
  }}>
        <label htmlFor="qwen38-ratio-length" style={labelStyle}>
          Average request length
          <input id="qwen38-ratio-length" type="number" min="1" step="1" value={requestLength} onChange={event => setRequestLength(event.target.value)} style={inputStyle} />
          <span style={{
    color: colors.muted,
    fontSize: "11px",
    fontWeight: 400
  }}>
            Input + output tokens — the ratio's only free parameter
          </span>
        </label>

        <label htmlFor="qwen38-ratio-concurrency" style={labelStyle}>
          Target concurrency
          <input id="qwen38-ratio-concurrency" type="number" min="1" step="1" value={targetConcurrency} onChange={event => setTargetConcurrency(event.target.value)} style={inputStyle} />
          <span style={{
    color: colors.muted,
    fontSize: "11px",
    fontWeight: 400
  }}>
            Requests in flight — sizes the explicit pin, not the ratio
          </span>
        </label>

        <div style={{
    display: "flex",
    flexDirection: "column",
    gap: "6px"
  }}>
          <span style={{
    fontSize: "12px",
    fontWeight: 600
  }}>
            Serving configuration (follows the Deploy panel and Playground)
          </span>
          <div style={{
    display: "flex",
    flexWrap: "wrap",
    gap: "6px"
  }}>
            {derivedChips.map(c => <span key={c} style={chipStyle}>{c}</span>)}
          </div>
        </div>
      </div>

      {!valid ? <div style={{
    color: colors.error,
    fontSize: "12px"
  }}>
          {tp !== 1 ? `This calculator models the single-GPU (TP1) geometry; the panels are at TP${tp}.` : "Enter a valid request length."}
        </div> : <div style={{
    display: "flex",
    alignItems: "center",
    gap: "12px",
    paddingTop: "12px",
    borderTop: `1px solid ${colors.border}`,
    flexWrap: "wrap"
  }}>
          <div>
            <div style={{
    color: colors.muted,
    fontSize: "11px"
  }}>
              Balanced ratio — pinned into the commands above
            </div>
            <div style={{
    fontSize: "26px",
    fontWeight: 700
  }}>{ratioStr}</div>
            {baseValid && baseRatioStr !== ratioStr ? <div style={{
    color: colors.muted,
    fontSize: "11px"
  }}>
                Deploy command (without Playground overrides): {baseRatioStr}
              </div> : null}
          </div>
          <code style={{
    flex: 1,
    minWidth: "240px",
    color: colors.text
  }}>
            {flagText}
          </code>
          <button type="button" onClick={copy} style={{
    padding: "7px 11px",
    border: 0,
    borderRadius: "5px",
    background: colors.accent,
    color: "#ffffff",
    fontSize: "12px",
    fontWeight: 600,
    cursor: "pointer"
  }}>
            {copied ? "Copied" : "Copy flag"}
          </button>
        </div>}
      <div style={{
    color: colors.muted,
    fontSize: "11px"
  }}>
        state/slot {(stateBytesPerSlot / 1e6).toFixed(1)} MB · KV/token{" "}
        {(kvBytesPerToken / 1e3).toFixed(1)} KB · the ratio prices{" "}
        {slots + drafts} state slots per request; the pin counts S = {slots},
        so {targetConcurrency || "N"} concurrent requests pin{" "}
        {pinValid ? pin : "—"} slots.
      </div>
    </div>;
};

export const config = {
  modelName: "Qwen3.8-27B",
  supportedHardware: ["h200", "rtx6000", "rtx5090", "dgx-spark", "gb300"],
  hardware: [{
    id: "rtx6000",
    label: "RTX PRO 6000",
    vram: "96GB",
    vendor: "blackwell"
  }, {
    id: "rtx5090",
    label: "RTX 5090",
    vram: "32GB",
    vendor: "blackwell"
  }],
  matchDims: [{
    id: "variant",
    title: "Model Variant",
    options: [{
      id: "default",
      label: "Default"
    }]
  }, {
    id: "quant",
    title: "Quantization",
    options: [{
      id: "bf16",
      label: "BF16"
    }, {
      id: "fp8",
      label: "FP8"
    }, {
      id: "nvfp4-bf16-head",
      label: "NVFP4-BF16-Head"
    }, {
      id: "nvfp4-fp4-head",
      label: "NVFP4-FP4-Head"
    }, {
      id: "nvfp4-nvidia",
      label: "NVFP4-NVIDIA"
    }]
  }, {
    id: "nodes",
    title: "Nodes",
    options: [{
      id: "single",
      label: "Single Node"
    }]
  }],
  overlayDims: [{
    id: "spec",
    title: "Speculative Decoding",
    default: "none",
    options: [{
      id: "none",
      label: "None"
    }, {
      id: "eagle",
      label: "EAGLE",
      disabled: sel => sel.hw === "rtx5090" && !String(sel.quant).startsWith("nvfp4"),
      disableReason: "On the 32GB RTX 5090 the MTP head only fits on top of the NVFP4 weights",
      stripPrefixes: sel => sel.hw === "rtx5090" ? ["--mem-fraction-static", "--mamba-full-memory-ratio", "--max-total-tokens"] : [],
      flags: sel => ["--speculative-algorithm EAGLE", "--speculative-num-steps 3", "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4", ...["rtx5090", "rtx6000", "dgx-spark"].includes(sel.hw) ? ["--enable-linear-replayssm-spec"] : [], ...sel.hw === "rtx5090" ? [sel.ssmDtype === "float32" ? "--mem-fraction-static 0.94" : "--mem-fraction-static 0.93"] : [], ...sel.hw === "rtx5090" && sel.quant === "nvfp4-bf16-head" && sel.ssmDtype === "float32" ? ["--max-total-tokens 16384", sel.tier === "low-latency" ? "--mamba-full-memory-ratio 2.67" : "--mamba-full-memory-ratio 2.14"] : []]
    }, {
      id: "dspark",
      label: "DSPARK",
      disabled: sel => sel.hw === "rtx5090" && !String(sel.quant).startsWith("nvfp4"),
      disableReason: "On the 32GB RTX 5090 the DSpark draft model only fits on top of the NVFP4 weights",
      stripPrefixes: sel => sel.hw === "rtx5090" ? ["--mem-fraction-static", "--mamba-full-memory-ratio", "--chunked-prefill-size", "--max-total-tokens"] : [],
      flags: sel => ["--speculative-algorithm DSPARK", "--speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark", "--speculative-draft-attention-backend flashinfer", ...sel.hw === "rtx5090" ? ["--max-total-tokens 16384", ...sel.quant === "nvfp4-bf16-head" ? ["--mem-fraction-static 0.92", "--chunked-prefill-size 512", sel.tier === "low-latency" ? "--mamba-full-memory-ratio 3.38" : "--mamba-full-memory-ratio 2.71"] : sel.ssmDtype === "float32" ? ["--mem-fraction-static 0.91", "--chunked-prefill-size 1024", sel.tier === "low-latency" ? "--mamba-full-memory-ratio 6.94" : "--mamba-full-memory-ratio 5.56"] : ["--mem-fraction-static 0.88", sel.tier === "low-latency" ? "--mamba-full-memory-ratio 3.38" : "--mamba-full-memory-ratio 3.12"]] : []]
    }, {
      id: "dflash",
      label: "DFLASH2",
      disabled: sel => sel.hw === "rtx5090" && !String(sel.quant).startsWith("nvfp4"),
      disableReason: "On the 32GB RTX 5090 the DFlash2 draft model only fits on top of the NVFP4 weights",
      stripPrefixes: sel => sel.hw === "rtx5090" ? sel.ssmDtype === "float32" ? ["--mem-fraction-static", "--mamba-full-memory-ratio"] : ["--mem-fraction-static"] : [],
      flags: sel => ["--speculative-algorithm DFLASH", "--speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2", "--speculative-num-draft-tokens 8", ...sel.hw === "rtx5090" ? sel.ssmDtype === "float32" ? ["--mem-fraction-static 0.895", "--mamba-full-memory-ratio 10"] : ["--mem-fraction-static 0.91", "--chunked-prefill-size 1024"] : [], ...sel.hw === "rtx5090" && sel.quant === "nvfp4-bf16-head" ? ["--max-total-tokens 16384", sel.tier === "low-latency" ? "--mamba-full-memory-ratio 3.38" : "--mamba-full-memory-ratio 3.12"] : []]
    }]
  }, {
    id: "tier",
    title: "Serving Strategy",
    default: "low-latency",
    stripPrefixes: ["--mamba-radix-cache-strategy"],
    options: [{
      id: "low-latency",
      label: "Low-Latency",
      flags: ["--mamba-radix-cache-strategy extra_buffer"]
    }, {
      id: "high-throughput",
      label: "High-Throughput",
      flags: ["--mamba-radix-cache-strategy extra_buffer_lazy"]
    }]
  }, {
    id: "ssmDtype",
    title: "Mamba SSM Dtype",
    default: "float32",
    options: [{
      id: "float32",
      label: "float32",
      disabled: sel => sel.hw === "rtx5090" && (sel.quant === "nvfp4-bf16-head" ? sel.spec === "dflash" || sel.spec === "dspark" : sel.spec === "dflash" && sel.tier === "low-latency"),
      disableReason: "On the 32GB RTX 5090 this combination has no fp32 GDN state pool that " + "also leaves room for prefill graph capture — use bfloat16",
      flags: ["--mamba-ssm-dtype float32"]
    }, {
      id: "bfloat16",
      label: "bfloat16",
      disabled: sel => sel.hw === "rtx5090" && !String(sel.quant).startsWith("nvfp4"),
      disableReason: "On the 32GB RTX 5090 the bf16 GDN state pool is only a live choice for NVFP4; " + "the BF16 and FP8 checkpoints have no serviceable cell on this card",
      flags: ["--mamba-ssm-dtype bfloat16"]
    }]
  }],
  modelNames: {
    "default|bf16": "Qwen/Qwen3.8-27B",
    "default|fp8": "Qwen/Qwen3.8-27B-FP8",
    "default|nvfp4-bf16-head": "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead",
    "default|nvfp4-fp4-head": "RadixArk/Qwen3.8-27B-NVFP4",
    "default|nvfp4-nvidia": "nvidia/Qwen3.8-27B-NVFP4"
  },
  placeholders: {
    HOST_IP: {
      target: "command",
      label: "Bind host",
      default: "0.0.0.0"
    },
    PORT: {
      target: "command",
      label: "Bind port",
      default: "30000"
    },
    HF_TOKEN: {
      target: "command",
      label: "HF token (Docker)",
      default: "<your-hf-token>"
    },
    CURL_HOST: {
      target: "curl",
      label: "Server host",
      default: "localhost"
    },
    CURL_PORT: {
      target: "curl",
      label: "Server port",
      default: "30000"
    }
  },
  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,
  benchmarkCommands: {
    speed: `python3 -m sglang.bench_serving \\
  --backend sglang-oai \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} --random-range-ratio 1 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --request-rate inf \\
  --flush-cache`,
    accuracy: {
      gsm8k_pct: `python3 -m sglang.test.run_eval \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --eval-name gsm8k \\
  --num-examples 1319`
    }
  },
  accuracyLabels: [["gsm8k_pct", "GSM8K", "%"]],
  dockerImages: {
    h200: "lmsysorg/sglang:latest",
    rtx6000: "lmsysorg/sglang:latest",
    rtx5090: "lmsysorg/sglang:latest",
    "dgx-spark": "lmsysorg/sglang:latest",
    gb300: "lmsysorg/sglang:latest"
  },
  github: {
    cookbookModel: "Qwen/Qwen3.8-27B"
  },
  playgroundFeatures: {
    parsers: {
      items: [{
        id: "reasoning",
        label: "Reasoning Parser",
        flag: "--reasoning-parser qwen3"
      }, {
        id: "toolCall",
        label: "Tool Call Parser",
        flag: "--tool-call-parser qwen3_coder"
      }]
    },
    speculative: {
      options: [{
        id: "current",
        label: "Inherited from base"
      }, {
        id: "off",
        label: "Off (greedy)"
      }, {
        id: "mtp",
        label: "EAGLE / MTP",
        flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3", "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"]
      }, {
        id: "dspark",
        label: "DSpark",
        flags: ["--speculative-algorithm DSPARK", "--speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark", "--speculative-draft-attention-backend flashinfer"]
      }, {
        id: "dflash",
        label: "DFlash2",
        flags: ["--speculative-algorithm DFLASH", "--speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2", "--speculative-num-draft-tokens 8"]
      }]
    },
    flagSelects: [{
      id: "attnBackend",
      title: "Attention Backend",
      stripPrefixes: ["--attention-backend"],
      options: [{
        id: "flashinfer",
        label: "FlashInfer (default)",
        flags: ["--attention-backend flashinfer"]
      }, {
        id: "fa3",
        label: "FlashAttention-3 (SM90 only)",
        flags: ["--attention-backend fa3"]
      }, {
        id: "triton",
        label: "Triton — MTP fallback on older FlashInfer",
        flags: ["--attention-backend triton"]
      }]
    }, {
      id: "kvCacheDtype",
      title: "KV Cache Precision",
      stripPrefixes: ["--kv-cache-dtype"],
      options: [{
        id: "auto",
        label: "Auto (checkpoint-declared)"
      }, {
        id: "fp8",
        label: "FP8 (E4M3) — halves KV memory",
        flags: ["--kv-cache-dtype fp8_e4m3"]
      }, {
        id: "bf16",
        label: "BFloat16",
        flags: ["--kv-cache-dtype bfloat16"]
      }]
    }, {
      id: "mambaSsmDtype",
      title: "GDN State Precision",
      stripPrefixes: ["--mamba-ssm-dtype"],
      options: [{
        id: "auto",
        label: "Auto (FP32)"
      }, {
        id: "bf16",
        label: "BFloat16 — halves state memory",
        flags: ["--mamba-ssm-dtype bfloat16"]
      }]
    }, {
      id: "prefixCache",
      title: "Prefix Cache",
      stripPrefixes: ["--disable-radix-cache"],
      options: [{
        id: "on",
        label: "On"
      }, {
        id: "off",
        label: "Off (S=1)",
        flags: ["--disable-radix-cache"]
      }]
    }, {
      id: "mambaRadix",
      title: "GDN Radix Cache Strategy",
      showWhen: (b, v, d) => ((v && v.prefixCache) ?? (d && d.prefixCache)) !== "off",
      stripPrefixes: ["--mamba-radix-cache-strategy"],
      options: [{
        id: "auto",
        label: "Auto (extra_buffer, S=5)"
      }, {
        id: "lazy",
        label: "extra_buffer_lazy (S=4)",
        flags: ["--mamba-radix-cache-strategy extra_buffer_lazy"]
      }, {
        id: "nobuf",
        label: "no_buffer (S=3)",
        flags: ["--mamba-radix-cache-strategy no_buffer"]
      }]
    }]
  },
  cells: [{
    match: {
      hw: "h200",
      variant: "default",
      quant: "fp8",
      nodes: "single"
    },
    verified: true,
    verificationStatus: sel => sel.spec === "dflash" ? "in-progress" : "verified",
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.85", "--attention-backend flashinfer", "--chunked-prefill-size 32768", "--max-prefill-tokens 32768", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "h200",
      variant: "default",
      quant: "bf16",
      nodes: "single"
    },
    verified: true,
    verificationStatus: sel => sel.spec === "dflash" ? "in-progress" : "verified",
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.85", "--attention-backend flashinfer", "--chunked-prefill-size 32768", "--max-prefill-tokens 32768", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "rtx6000",
      variant: "default",
      quant: "nvfp4-bf16-head",
      nodes: "single"
    },
    verified: true,
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.85", "--attention-backend flashinfer", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "rtx6000",
      variant: "default",
      quant: "nvfp4-fp4-head",
      nodes: "single"
    },
    verified: true,
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.85", "--attention-backend flashinfer", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "rtx6000",
      variant: "default",
      quant: "nvfp4-nvidia",
      nodes: "single"
    },
    verified: true,
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.85", "--attention-backend flashinfer", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "rtx6000",
      variant: "default",
      quant: "fp8",
      nodes: "single"
    },
    verified: true,
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.85", "--attention-backend flashinfer", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "rtx6000",
      variant: "default",
      quant: "bf16",
      nodes: "single"
    },
    verified: true,
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.85", "--attention-backend flashinfer", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "rtx5090",
      variant: "default",
      quant: "nvfp4-bf16-head",
      nodes: "single"
    },
    verified: true,
    warn: "This recipe serves ONE request at a time: --max-running-requests 1 " + "and --cuda-graph-max-bs-decode 1 pin it to the validated single-stream " + "envelope. To handle more concurrent requests, raise both flags " + "together and re-derive --mamba-full-memory-ratio (and mem-fraction) " + "with the [Mamba ratio calculator](#mamba-ratio-calculator) — on this " + "32GB card the GDN state pool, not KV, is what runs out first.",
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.9", "--attention-backend flashinfer", "--max-running-requests 1", "--cuda-graph-max-bs-decode 1", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "rtx5090",
      variant: "default",
      quant: "nvfp4-fp4-head",
      nodes: "single"
    },
    verified: true,
    warn: "This recipe serves ONE request at a time: --max-running-requests 1 " + "and --cuda-graph-max-bs-decode 1 pin it to the validated single-stream " + "envelope. To handle more concurrent requests, raise both flags " + "together and re-derive --mamba-full-memory-ratio (and mem-fraction) " + "with the [Mamba ratio calculator](#mamba-ratio-calculator) — on this " + "32GB card the GDN state pool, not KV, is what runs out first.",
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.9", "--attention-backend flashinfer", "--max-running-requests 1", "--cuda-graph-max-bs-decode 1", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "rtx5090",
      variant: "default",
      quant: "nvfp4-nvidia",
      nodes: "single"
    },
    verified: true,
    warn: "This recipe serves ONE request at a time: --max-running-requests 1 " + "and --cuda-graph-max-bs-decode 1 pin it to the validated single-stream " + "envelope. To handle more concurrent requests, raise both flags " + "together and re-derive --mamba-full-memory-ratio (and mem-fraction) " + "with the [Mamba ratio calculator](#mamba-ratio-calculator) — on this " + "32GB card the GDN state pool, not KV, is what runs out first.",
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.9", "--attention-backend flashinfer", "--max-running-requests 1", "--cuda-graph-max-bs-decode 1", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "dgx-spark",
      variant: "default",
      quant: "nvfp4-bf16-head",
      nodes: "single"
    },
    verified: true,
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.80", "--attention-backend flashinfer", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "dgx-spark",
      variant: "default",
      quant: "nvfp4-fp4-head",
      nodes: "single"
    },
    verified: true,
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.80", "--attention-backend flashinfer", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "dgx-spark",
      variant: "default",
      quant: "nvfp4-nvidia",
      nodes: "single"
    },
    verified: true,
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.80", "--attention-backend flashinfer", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "dgx-spark",
      variant: "default",
      quant: "fp8",
      nodes: "single"
    },
    verified: true,
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.80", "--attention-backend flashinfer", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "dgx-spark",
      variant: "default",
      quant: "bf16",
      nodes: "single"
    },
    verified: true,
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.80", "--attention-backend flashinfer", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "gb300",
      variant: "default",
      quant: "nvfp4-bf16-head",
      nodes: "single"
    },
    verified: true,
    verificationStatus: sel => sel.spec === "dflash" ? "in-progress" : "verified",
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.85", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "gb300",
      variant: "default",
      quant: "nvfp4-fp4-head",
      nodes: "single"
    },
    verified: true,
    verificationStatus: sel => sel.spec === "dflash" ? "in-progress" : "verified",
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.85", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "gb300",
      variant: "default",
      quant: "nvfp4-nvidia",
      nodes: "single"
    },
    verified: true,
    verificationStatus: sel => sel.spec === "dflash" ? "in-progress" : "verified",
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.85", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "gb300",
      variant: "default",
      quant: "fp8",
      nodes: "single"
    },
    verified: true,
    verificationStatus: sel => sel.spec === "dflash" ? "in-progress" : "verified",
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.85", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }, {
    match: {
      hw: "gb300",
      variant: "default",
      quant: "bf16",
      nodes: "single"
    },
    verified: true,
    verificationStatus: sel => sel.spec === "dflash" ? "in-progress" : "verified",
    env: [],
    flags: ["--trust-remote-code", "--model-path {{MODEL_NAME}}", "--kv-cache-dtype fp8_e4m3", "--mem-fraction-static 0.85", "--chunked-prefill-size 2048", "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--host {{HOST_IP}}", "--port {{PORT}}"]
  }]
};

export const Deployment = ({config, benchmarks}) => {
  if (!config) {
    return <div style={{
      padding: 12,
      color: "#b91c1c"
    }}>Deployment: missing <code>config</code> prop</div>;
  }
  const AMD_RDMA_DOCKER_FLAGS = ["--device /dev/infiniband", "--cap-add IPC_LOCK", "--ulimit memlock=-1", "--ulimit stack=67108864", "--ulimit nofile=1048576:1048576"];
  const HARDWARE_CATALOG = {
    blackwell: [{
      id: "b300",
      label: "B300",
      vram: "288GB"
    }, {
      id: "gb300",
      label: "GB300",
      vram: "288GB"
    }, {
      id: "b200",
      label: "B200",
      vram: "192GB"
    }, {
      id: "gb200",
      label: "GB200",
      vram: "192GB"
    }, {
      id: "dgx-spark",
      label: "DGX Spark",
      vram: "128GB",
      multiNodeDockerFlags: ["--ulimit memlock=-1:-1", "--cap-add IPC_LOCK", "--device /dev/infiniband"]
    }],
    hopper: [{
      id: "h200",
      label: "H200",
      vram: "141GB"
    }, {
      id: "h100",
      label: "H100",
      vram: "80GB"
    }, {
      id: "h20-3e",
      label: "H20-3e",
      vram: "141GB"
    }, {
      id: "h800",
      label: "H800",
      vram: "80GB"
    }],
    amd: [{
      id: "mi300x",
      label: "MI300X",
      vram: "192GB",
      multiNodeDockerFlags: [...AMD_RDMA_DOCKER_FLAGS]
    }, {
      id: "mi325x",
      label: "MI325X",
      vram: "256GB",
      multiNodeDockerFlags: [...AMD_RDMA_DOCKER_FLAGS]
    }, {
      id: "mi350x",
      label: "MI350X",
      vram: "288GB",
      multiNodeDockerFlags: [...AMD_RDMA_DOCKER_FLAGS]
    }, {
      id: "mi355x",
      label: "MI355X",
      vram: "288GB",
      multiNodeDockerFlags: [...AMD_RDMA_DOCKER_FLAGS]
    }],
    npu: [{
      id: "a3",
      label: "A3 Series",
      vram: "64GB/die",
      npuDevices: 16
    }, {
      id: "a5",
      label: "950PR/DT Series",
      vram: "128GB",
      npuDevices: 8
    }]
  };
  const makeStyles = isDark => ({
    container: {
      maxWidth: "900px",
      margin: "0 auto",
      display: "flex",
      flexDirection: "column",
      gap: "3px"
    },
    card: {
      padding: "5px 10px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#E85D4D" : "#D45D44"}`,
      borderRadius: "4px",
      display: "flex",
      alignItems: "center",
      gap: "10px",
      background: isDark ? "#1f2937" : "#fff"
    },
    cardColumn: {
      padding: "5px 10px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#E85D4D" : "#D45D44"}`,
      borderRadius: "4px",
      display: "flex",
      flexDirection: "column",
      gap: "4px",
      background: isDark ? "#1f2937" : "#fff"
    },
    title: {
      fontSize: "12px",
      fontWeight: "600",
      minWidth: "108px",
      flexShrink: 0,
      color: isDark ? "#e5e7eb" : "inherit"
    },
    vendorRow: {
      display: "flex",
      alignItems: "center",
      gap: "6px"
    },
    vendorLabel: {
      fontSize: "10px",
      fontWeight: "600",
      color: isDark ? "#9ca3af" : "#6b7280",
      width: "68px",
      flexShrink: 0,
      textTransform: "uppercase",
      letterSpacing: "0.04em"
    },
    itemsGrid: () => ({
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(72px, 1fr))",
      gap: "4px",
      flex: 1
    }),
    labelBase: {
      padding: "2px 8px",
      border: `1px solid ${isDark ? "#9ca3af" : "#d1d5db"}`,
      borderRadius: "3px",
      cursor: "pointer",
      display: "inline-flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      fontWeight: "500",
      fontSize: "12px",
      transition: "all 0.2s",
      userSelect: "none",
      minHeight: "26px",
      textAlign: "center",
      background: isDark ? "#374151" : "#fff",
      color: isDark ? "#e5e7eb" : "inherit"
    },
    checked: {
      background: "#D45D44",
      color: "white",
      borderColor: "#D45D44"
    },
    disabled: {
      cursor: "not-allowed",
      opacity: 0.4
    },
    subtitle: {
      display: "block",
      fontSize: "9px",
      marginTop: "1px",
      lineHeight: "1.1",
      opacity: 0.7
    },
    commandWrap: {
      position: "relative",
      flex: 1,
      background: isDark ? "#111827" : "#f5f5f5",
      borderRadius: "6px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      overflow: "hidden"
    },
    commandHeader: {
      display: "flex",
      flexWrap: "wrap",
      justifyContent: "space-between",
      alignItems: "center",
      gap: "6px 10px",
      padding: "6px 10px",
      borderBottom: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      background: isDark ? "#1f2937" : "#fafafa"
    },
    commandPre: {
      padding: "12px 16px",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
      fontSize: "12px",
      lineHeight: "1.5",
      color: isDark ? "#e5e7eb" : "#374151",
      whiteSpace: "pre-wrap",
      overflowX: "auto",
      margin: 0
    },
    mtpWarn: {
      margin: "8px 0 0",
      padding: "8px 12px",
      borderRadius: "8px",
      fontSize: "12px",
      lineHeight: "1.45",
      background: isDark ? "#78350f" : "#fef3c7",
      color: isDark ? "#fde68a" : "#92400e",
      border: `1px solid ${isDark ? "#92400e" : "#fcd34d"}`
    },
    badge: status => ({
      display: "inline-flex",
      alignItems: "center",
      gap: "6px",
      padding: "2px 8px",
      borderRadius: "10px",
      background: ({
        verified: isDark ? "#064e3b" : "#d1fae5",
        "in-progress": isDark ? "#1e3a8a" : "#dbeafe",
        unverified: isDark ? "#78350f" : "#fef3c7"
      })[verifyStatusOf(status)],
      color: ({
        verified: isDark ? "#a7f3d0" : "#065f46",
        "in-progress": isDark ? "#bfdbfe" : "#1e40af",
        unverified: isDark ? "#fde68a" : "#92400e"
      })[verifyStatusOf(status)],
      fontSize: "11px",
      fontWeight: 600,
      whiteSpace: "nowrap"
    }),
    badgeDot: status => ({
      width: "8px",
      height: "8px",
      borderRadius: "50%",
      background: ({
        verified: "#10b981",
        "in-progress": "#3b82f6",
        unverified: "#f59e0b"
      })[verifyStatusOf(status)]
    }),
    iconButton: {
      padding: "4px 10px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
      color: isDark ? "#e5e7eb" : "#374151",
      fontSize: "11px",
      fontWeight: 500,
      cursor: "pointer",
      display: "inline-flex",
      alignItems: "center",
      gap: "4px"
    },
    iconRow: {
      display: "inline-flex",
      flexWrap: "wrap",
      gap: "6px"
    },
    runModeWrap: {
      display: "inline-flex",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "10px",
      overflow: "hidden",
      fontSize: "11px",
      fontWeight: 600,
      userSelect: "none"
    },
    runModeChip: active => ({
      padding: "2px 10px",
      cursor: "pointer",
      background: active ? isDark ? "#1f2937" : "#fff" : "transparent",
      color: active ? isDark ? "#e5e7eb" : "#111827" : isDark ? "#9ca3af" : "#6b7280",
      borderRight: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`
    }),
    runModeChipLast: active => ({
      padding: "2px 10px",
      cursor: "pointer",
      background: active ? isDark ? "#1f2937" : "#fff" : "transparent",
      color: active ? isDark ? "#e5e7eb" : "#111827" : isDark ? "#9ca3af" : "#6b7280"
    }),
    headerLeft: {
      display: "inline-flex",
      flexWrap: "wrap",
      alignItems: "center",
      gap: "8px"
    },
    modalBackdrop: {
      position: "fixed",
      inset: 0,
      background: "rgba(0,0,0,0.5)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 9999
    },
    modalBox: {
      background: isDark ? "#1f2937" : "#fff",
      color: isDark ? "#e5e7eb" : "#111827",
      borderRadius: "8px",
      padding: "20px",
      maxWidth: "720px",
      width: "92%",
      maxHeight: "85vh",
      overflowY: "auto",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      boxShadow: "0 10px 25px rgba(0,0,0,0.25)"
    },
    modalHeader: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      marginBottom: "12px"
    },
    modalTitle: {
      fontSize: "15px",
      fontWeight: 600
    },
    modalCloseBtn: {
      background: "transparent",
      border: "none",
      color: "inherit",
      fontSize: "20px",
      cursor: "pointer",
      padding: "0 6px",
      lineHeight: 1
    },
    formField: {
      display: "flex",
      flexDirection: "column",
      gap: "4px",
      marginBottom: "10px"
    },
    formLabel: {
      fontSize: "12px",
      fontWeight: 500,
      color: isDark ? "#9ca3af" : "#4b5563"
    },
    formInput: {
      padding: "6px 10px",
      fontSize: "13px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "4px",
      background: isDark ? "#111827" : "#fff",
      color: isDark ? "#e5e7eb" : "#111827",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace"
    },
    sectionHeading: {
      fontSize: "12px",
      fontWeight: 600,
      textTransform: "uppercase",
      letterSpacing: "0.04em",
      color: isDark ? "#9ca3af" : "#6b7280",
      margin: "12px 0 6px 0"
    },
    primaryBtn: {
      padding: "6px 14px",
      background: "#D45D44",
      color: "white",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      fontSize: "13px",
      fontWeight: 500
    },
    benchCard: {
      padding: "8px 12px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#E85D4D" : "#D45D44"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
      display: "flex",
      flexDirection: "column",
      gap: "8px"
    },
    benchHeader: {
      display: "flex",
      flexWrap: "wrap",
      alignItems: "baseline",
      justifyContent: "space-between",
      gap: "6px 12px"
    },
    benchTitle: {
      fontSize: "13px",
      fontWeight: 600,
      color: isDark ? "#e5e7eb" : "inherit"
    },
    benchVersion: {
      fontSize: "11px",
      color: isDark ? "#9ca3af" : "#6b7280"
    },
    benchHeaderRight: {
      display: "flex",
      flexWrap: "wrap",
      alignItems: "center",
      gap: "6px 10px",
      flexShrink: 0
    },
    benchChipRow: {
      display: "flex",
      alignItems: "center",
      gap: "6px",
      flexWrap: "wrap",
      margin: "2px 0 8px"
    },
    benchChip: {
      padding: "2px 10px",
      fontSize: "12px",
      cursor: "pointer",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
      color: isDark ? "#e5e7eb" : "#374151",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace"
    },
    benchChipActive: {
      background: "#D45D44",
      color: "white",
      borderColor: "#D45D44"
    },
    benchBlock: {
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderRadius: "4px",
      padding: "8px 10px",
      background: isDark ? "#111827" : "#fafafa"
    },
    benchBlockTitle: {
      fontSize: "11px",
      fontWeight: 600,
      textTransform: "uppercase",
      letterSpacing: "0.04em",
      color: isDark ? "#9ca3af" : "#6b7280",
      marginBottom: "4px"
    },
    benchWorkload: {
      fontSize: "11px",
      fontStyle: "italic",
      color: isDark ? "#9ca3af" : "#6b7280",
      marginBottom: "6px",
      lineHeight: "1.3"
    },
    benchRow: {
      display: "flex",
      justifyContent: "space-between",
      fontSize: "12px",
      padding: "2px 0"
    },
    benchKey: {
      color: isDark ? "#9ca3af" : "#6b7280"
    },
    benchVal: {
      color: isDark ? "#e5e7eb" : "#111827",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
      fontWeight: 500
    },
    benchNotes: {
      fontSize: "11px",
      fontStyle: "italic",
      color: isDark ? "#9ca3af" : "#6b7280"
    },
    benchLegend: {
      fontSize: "10px",
      fontStyle: "italic",
      color: isDark ? "#6b7280" : "#9ca3af",
      marginTop: "6px",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace"
    },
    benchEmpty: {
      fontSize: "12px",
      fontStyle: "italic",
      color: isDark ? "#9ca3af" : "#6b7280"
    },
    benchTableScroll: {
      overflowX: "auto",
      overflowY: "hidden",
      marginTop: "4px"
    },
    benchTable: {
      display: "grid",
      columnGap: 0,
      rowGap: "3px",
      alignItems: "baseline",
      minWidth: "max-content"
    },
    benchTableHead: {
      textAlign: "right",
      fontWeight: 500,
      fontSize: "11px",
      color: isDark ? "#9ca3af" : "#6b7280",
      paddingLeft: "16px",
      paddingBottom: "4px",
      whiteSpace: "nowrap"
    },
    benchTableCornerHead: {
      paddingBottom: "4px",
      position: "sticky",
      left: 0,
      background: isDark ? "#111827" : "#fafafa",
      zIndex: 1,
      alignSelf: "stretch"
    },
    benchTableSeparator: {
      gridColumn: "1 / -1",
      height: "1px",
      background: isDark ? "#374151" : "#e5e7eb",
      marginTop: "-3px"
    },
    benchTableLabel: {
      textAlign: "left",
      fontSize: "12px",
      color: isDark ? "#9ca3af" : "#6b7280",
      whiteSpace: "nowrap",
      position: "sticky",
      left: 0,
      background: isDark ? "#111827" : "#fafafa",
      zIndex: 1
    },
    benchTableValue: {
      textAlign: "right",
      fontSize: "12px",
      color: isDark ? "#e5e7eb" : "#111827",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
      fontWeight: 500,
      paddingLeft: "16px",
      whiteSpace: "nowrap"
    },
    benchTableValueMissing: {
      color: isDark ? "#6b7280" : "#9ca3af"
    }
  });
  const VERIFY_LABEL = {
    verified: "Verified",
    "in-progress": "Final Verification In Progress",
    unverified: "Not Verified"
  };
  const verifyStatusOf = v => typeof v === "string" ? VERIFY_LABEL[v] ? v : "unverified" : v ? "verified" : "unverified";
  const cellVerifyStatus = (c, sel) => {
    if (!c) return "unverified";
    const v = typeof c.verificationStatus === "function" ? c.verificationStatus(sel) : c.verificationStatus;
    return verifyStatusOf(v ?? c.verified);
  };
  const LEGACY_MATCH_DIMS = [{
    id: "variant",
    title: "Model Variant",
    optionsKey: "variants"
  }, {
    id: "quant",
    title: "Quantization",
    optionsKey: "quantizations"
  }, {
    id: "strategy",
    title: "Strategy",
    optionsKey: "strategies"
  }, {
    id: "nodes",
    title: "Nodes",
    optionsKey: "nodesOptions"
  }];
  const matchDimSpecs = (config.matchDims || LEGACY_MATCH_DIMS).map(d => ({
    ...d,
    options: d.options || config[d.optionsKey] || []
  }));
  const overlayDimSpecs = config.overlayDims || [];
  const commandBuilder = config.commandBuilder || null;
  const DIMENSIONS = ["hw", ...matchDimSpecs.map(d => d.id)];
  const optionVisible = (opt, sel) => typeof opt.showWhen !== "function" || opt.showWhen(sel);
  const optionDisabled = (opt, sel) => typeof opt.disabled === "function" ? opt.disabled(sel) : !!opt.disabled;
  const visibleOptions = (spec, sel) => (spec.options || []).filter(o => optionVisible(o, sel));
  const rowVisible = (spec, sel) => (typeof spec.showWhen !== "function" || spec.showWhen(sel)) && visibleOptions(spec, sel).length > 0;
  const overlayPick = sel => {
    const picked = [];
    for (const spec of config.overlayDims || []) {
      if (!rowVisible(spec, sel)) continue;
      const opt = (spec.options || []).find(o => o.id === sel[spec.id]);
      if (opt && !optionDisabled(opt, sel)) picked.push(opt);
    }
    return picked;
  };
  const overlayPart = (sel, key) => {
    const out = [];
    for (const opt of overlayPick(sel)) {
      const add = typeof opt[key] === "function" ? opt[key](sel) : opt[key];
      if (add) out.push(...add);
    }
    return out;
  };
  const overlayCompose = (cellFlags, sel) => {
    const strip = overlayPart(sel, "stripPrefixes");
    const add = overlayPart(sel, "flags");
    if (!strip.length) return [...cellFlags || [], ...add];
    const used = new Set();
    const replacementsFor = tok => {
      const out = [];
      add.forEach((f, i) => {
        if (used.has(i) || f.split(/[\s=]/)[0] !== tok) return;
        used.add(i);
        out.push(f);
      });
      return out;
    };
    const out = [];
    for (const f of cellFlags || []) {
      const tok = f.split(/[\s=]/)[0];
      if (!strip.includes(tok)) out.push(f); else out.push(...replacementsFor(tok));
    }
    add.forEach((f, i) => {
      if (!used.has(i)) out.push(f);
    });
    return out;
  };
  const optionSoft = (opt, sel) => typeof opt.soft === "function" ? opt.soft(sel) : !!opt.soft;
  const findCell = (cells, sel) => cells.find(c => DIMENSIONS.every(d => c.match[d] === sel[d]));
  const findBenchmark = (list, sel) => {
    const hits = (list || []).filter(b => Object.entries(b.match || ({})).every(([k, v]) => sel[k] === v));
    return hits.sort((a, b) => Object.keys(b.match).length - Object.keys(a.match).length)[0] || null;
  };
  const normalizeSpeed = speed => {
    if (!speed) return [];
    return Array.isArray(speed) ? speed : [speed];
  };
  const effectiveAccuracy = (entry, sel) => entry ? {
    ...config.defaultAccuracy && config.defaultAccuracy[sel.variant] || ({}),
    ...entry.accuracy || ({})
  } : {};
  const benchmarkIsEmpty = (entry, accuracy) => {
    for (const m of normalizeSpeed(entry && entry.speed)) {
      if (m && typeof m === "object") {
        for (const [key, v] of Object.entries(m)) {
          if (key === "workload") continue;
          if (v !== null && v !== undefined) return false;
        }
      }
    }
    if (accuracy && typeof accuracy === "object") {
      for (const v of Object.values(accuracy)) {
        if (v !== null && v !== undefined) return false;
      }
    }
    return true;
  };
  const isOptionAvailable = (cells, sel, dim, value) => {
    const idx = DIMENSIONS.indexOf(dim);
    const higher = DIMENSIONS.slice(0, idx);
    return cells.some(c => c.match[dim] === value && higher.every(d => c.match[d] === sel[d]));
  };
  const snapToValidCell = (cells, sel, dim, value) => {
    const idx = DIMENSIONS.indexOf(dim);
    const higher = DIMENSIONS.slice(0, idx);
    const lower = DIMENSIONS.slice(idx + 1);
    let best = null, bestLowerMatches = -1;
    for (const c of cells) {
      if (c.match[dim] !== value) continue;
      if (!higher.every(d => c.match[d] === sel[d])) continue;
      let s = 0;
      for (const d of lower) if (c.match[d] === sel[d]) s++;
      if (s > bestLowerMatches) {
        bestLowerMatches = s;
        best = c;
      }
    }
    if (!best) return sel;
    const next = {
      ...sel,
      [dim]: value
    };
    for (const d of lower) next[d] = best.match[d];
    return next;
  };
  const validateSelection = (cells, parsed) => {
    const valid = {};
    for (const dim of DIMENSIONS) {
      const want = parsed[dim];
      const works = cells.some(c => c.match[dim] === want && DIMENSIONS.slice(0, DIMENSIONS.indexOf(dim)).every(d => c.match[d] === valid[d]));
      if (works) {
        valid[dim] = want;
      } else {
        const fallback = cells.find(c => DIMENSIONS.slice(0, DIMENSIONS.indexOf(dim)).every(d => c.match[d] === valid[d]));
        valid[dim] = fallback ? fallback.match[dim] : want;
      }
    }
    for (const spec of overlayDimSpecs) {
      const want = parsed[spec.id];
      const opts = spec.options || [];
      const picked = opts.some(o => o.id === want) ? want : (spec.default ?? (opts[0] && opts[0].id)) ?? "";
      const withPick = {
        ...valid,
        [spec.id]: picked
      };
      const usable = visibleOptions(spec, withPick).filter(o => !optionDisabled(o, withPick));
      valid[spec.id] = usable.some(o => o.id === picked) ? picked : (usable[0] && usable[0].id) ?? picked;
    }
    return valid;
  };
  const resolveModelName = sel => {
    const keys = [`${sel.hw}|${sel.variant}|${sel.quant}`, `${sel.variant}|${sel.quant}`, `${sel.hw}|${sel.quant}`, sel.quant, sel.hw, "default"];
    for (const k of keys) {
      const hit = config.modelNames[k];
      if (hit) return hit;
    }
    return "";
  };
  const interpolate = (text, env, modelName) => text.replace(/{{(\w+)}}/g, (_, key) => key === "MODEL_NAME" ? modelName : env[key] ?? `{{${key}}}`);
  const parseNnodes = id => {
    if (Number.isInteger(id)) return id;
    if ((/^\d+$/).test(id || "")) return parseInt(id, 10);
    if (id === "single") return 1;
    const m = (/^multi-(\d+)$/).exec(id || "");
    return m ? parseInt(m[1], 10) : 1;
  };
  const cellNnodes = (cell, sel) => sel.nodes !== undefined ? parseNnodes(sel.nodes) : cell.nnodes || 1;
  const PD_SERVE_PORTS = {
    prefill: 30000,
    decode: 30100
  };
  const overlayEnv = sel => overlayPart(sel, "env");
  const overlayHints = sel => overlayPart(sel, "hints");
  const renderCommand = (cell, sel, envValues, mode = "python") => {
    if (!cell) return "# No command available for the current selection.";
    const modelName = resolveModelName(sel);
    const nnodes = cellNnodes(cell, sel);
    const multinode = nnodes > 1;
    const cellEnv = [...cell.env || [], ...overlayEnv(sel)];
    const flags = overlayCompose(cell.flags, sel);
    if (multinode) {
      const PARALLELISM_ANCHORS = new Set(["--enable-dp-attention", "--dp-size", "--dp", "--tp-size", "--tp", "--sp-degree", "--ulysses-degree", "--ring-degree"]);
      let i = flags.reduce((last, flag, index) => PARALLELISM_ANCHORS.has(flag.split(/[\s=]/)[0]) ? index : last, -1);
      if (i === -1) i = flags.findIndex(f => f.startsWith("--model-path"));
      flags.splice(i + 1, 0, `--nnodes ${nnodes}`, `--node-rank {{NODE_RANK}}`, `--dist-init-addr {{NODE0_IP}}:20000`);
    }
    const pdServePort = PD_SERVE_PORTS[sel.pdMode];
    if (pdServePort !== undefined) {
      for (let j = 0; j < flags.length; j++) {
        if (flags[j].split(/[\s=]/)[0] === "--port") {
          flags[j] = `--port ${pdServePort}`;
        }
      }
    }
    let cmd;
    if (mode === "docker") {
      const di = config.dockerImages || ({});
      const image = di[`${sel.hw}|${sel.variant}|${sel.quant}`] || di[`${sel.variant}|${sel.quant}`] || di[`${sel.hw}|${sel.quant}|${sel.strategy}`] || di[`${sel.hw}|${sel.quant}`] || di[sel.hw] || "lmsysorg/sglang:dev";
      const dockerRunCommand = typeof config.dockerRunCommand === "function" ? config.dockerRunCommand(sel) : config.dockerRunCommand || "sglang serve";
      const portFlag = flags.find(x => x.split(/[\s=]/)[0] === "--port");
      const servePort = portFlag ? portFlag.slice(("--port").length).trim() : "{{PORT}}";
      const hostNetwork = multinode || typeof config.dockerHostNetworkWhen === "function" && config.dockerHostNetworkWhen(sel, {
        flags,
        env: cellEnv
      });
      const vendorOf = hwId => {
        for (const [vendor, list] of Object.entries(HARDWARE_CATALOG)) {
          if (list.some(h => h.id === hwId)) return vendor;
        }
        const extra = (config.hardware || []).find(h => h.id === hwId);
        return extra && extra.vendor || "nvidia";
      };
      const catalogEntryOf = hwId => {
        const extra = (config.hardware || []).find(h => h.id === hwId);
        if (extra) return extra;
        for (const list of Object.values(HARDWARE_CATALOG)) {
          const hit = list.find(h => h.id === hwId);
          if (hit) return hit;
        }
        return null;
      };
      const fabricFlagsOf = hwId => (catalogEntryOf(hwId) || ({})).multiNodeDockerFlags || [];
      const davinciLines = devices => {
        const lines = [];
        for (let i = 0; i < devices; i += 4) {
          const group = [];
          for (let k = i; k < Math.min(i + 4, devices); k++) {
            group.push(`--device=/dev/davinci${k}`);
          }
          lines.push("  " + group.join(" "));
        }
        return lines;
      };
      const gpuAccessLines = vendorOf(sel.hw) === "amd" ? ["docker run", "  --device=/dev/kfd --device=/dev/dri", "  --group-add video", "  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined", "  --shm-size 32g"] : vendorOf(sel.hw) === "npu" ? ["docker run --privileged --shm-size=16g", ...davinciLines((catalogEntryOf(sel.hw) || ({})).npuDevices || 16), "  --device=/dev/davinci_manager", "  --device=/dev/hisi_hdc", "  -v /usr/local/sbin:/usr/local/sbin", "  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver", "  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware", "  -v /etc/ascend_install.info:/etc/ascend_install.info", "  -v /var/queue_schedule:/var/queue_schedule", "  -v ~/.cache/:/root/.cache/"] : ["docker run --gpus all", "  --shm-size 32g"];
      const dockerLines = [...gpuAccessLines, hostNetwork ? "  --network host" : `  -p ${servePort}:${servePort}`, ...multinode ? fabricFlagsOf(sel.hw).map(f => "  " + f) : [], ...vendorOf(sel.hw) === "npu" ? [] : ["  -v ~/.cache/huggingface:/root/.cache/huggingface"], ...(config.dockerMounts || []).map(mount => `  -v ${mount}`), ...config.placeholders && config.placeholders.HF_TOKEN ? [`  --env "HF_TOKEN={{HF_TOKEN}}"`] : [], ...cellEnv.map(e => `  --env ${e}`), "  --ipc=host", `  ${image}`, `  ${dockerRunCommand}`, ...flags.map(f => "    " + f)];
      cmd = dockerLines.join(" \\\n");
    } else {
      const flagBlock = flags.map(f => "  " + f).join(" \\\n");
      const envBlock = cellEnv.length ? cellEnv.join(" \\\n") + " \\\n" : "";
      cmd = `${envBlock}sglang serve \\\n${flagBlock}`;
    }
    const hintLines = [...overlayHints(sel), ...multinode && config.multiNodeHints && config.multiNodeHints[sel.hw] ? config.multiNodeHints[sel.hw] : []];
    if (hintLines.length) {
      const hint = hintLines.map(line => line.length ? "# " + line : "#").join("\n");
      cmd = `${hint}\n${cmd}`;
    }
    cmd = interpolate(cmd, envValues, modelName);
    if (multinode) {
      const header = `# Multi-node (${nnodes} nodes). Run the same command on every node with:\n` + `#   <node-rank> = 0 on the head node, 1..${nnodes - 1} on the others\n` + `#   <node0-ip>  = IP of the head node (reachable from all others)`;
      cmd = `${header}\n${cmd}`;
    }
    return cmd;
  };
  const ACCURACY_LABELS = config.accuracyLabels || [];
  const renderBenchmarkCard = entry => {
    const pct = entry && entry.latencyPercentile || config.latencyPercentile || "P50";
    const SPEED_LABELS = [["ttft_ms", `TTFT (${pct})`, "ms"], ["tpot_ms", `TPOT (${pct})`, "ms"], ["tokens_per_sec_per_gpu", "throughput per gpu", "tok/s"], ["interactivity", "interactivity", "tokens/s/user", m => m.tpot_ms != null && m.tpot_ms !== 0 ? Math.round(1000 / m.tpot_ms * 10) / 10 : null]];
    const WORKLOAD_KEYS = ["dataset", "isl", "osl", "max_concurrency"];
    const fmt = (val, unit) => {
      if (val === null || val === undefined) return null;
      return `${val}${unit ? " " + unit : ""}`;
    };
    const formatWorkloadParts = (workload, keys) => {
      if (!workload) return "";
      const parts = [];
      if (keys.has("dataset") && workload.dataset) parts.push(workload.dataset);
      if (keys.has("isl") || keys.has("osl")) {
        if (workload.isl != null || workload.osl != null) {
          parts.push(`in/out=${workload.isl != null ? workload.isl : "?"}/${workload.osl != null ? workload.osl : "?"}`);
        }
      }
      if (keys.has("max_concurrency") && workload.max_concurrency != null) {
        parts.push(`max-concurrency=${workload.max_concurrency}`);
      }
      return parts.join(", ");
    };
    const ALWAYS_PER_COLUMN = new Set(["max_concurrency"]);
    const ATOMIC_WORKLOAD_GROUPS = [["isl", "osl"]];
    const partitionWorkload = measurements => {
      const shared = new Set();
      const differing = new Set();
      for (const k of WORKLOAD_KEYS) {
        const seen = new Set();
        let anyPresent = false;
        for (const m of measurements) {
          const v = m && m.workload ? m.workload[k] : undefined;
          if (v != null) anyPresent = true;
          seen.add(v);
        }
        if (!anyPresent) continue;
        if (ALWAYS_PER_COLUMN.has(k) || seen.size > 1) differing.add(k); else shared.add(k);
      }
      for (const group of ATOMIC_WORKLOAD_GROUPS) {
        if (!group.some(k => differing.has(k))) continue;
        for (const k of group) if (shared.delete(k)) differing.add(k);
      }
      return {
        shared,
        differing
      };
    };
    const renderBenchTable = ({title, sharedText, colHeaders, rows, colCount, legend}) => {
      if (rows.length === 0) return null;
      const showColHeaders = colHeaders.length > 0 && colHeaders.some(h => h !== "");
      return <div style={s.benchBlock}>
          <div style={s.benchBlockTitle}>{title}</div>
          {sharedText && <div style={s.benchWorkload}>{sharedText}</div>}
          {}
          <div className="sg-bench-scroll" style={s.benchTableScroll} {...colCount > 1 ? {
        tabIndex: 0,
        role: "group",
        "aria-label": sharedText ? `${title} — ${sharedText}` : title
      } : {}}>
            <div style={{
        ...s.benchTable,
        gridTemplateColumns: `max-content repeat(${colCount}, minmax(max-content, 1fr))`
      }}>
              {showColHeaders && <div key="corner" style={s.benchTableCornerHead}></div>}
              {showColHeaders && colHeaders.map((h, i) => <div key={`hdr-${i}`} style={s.benchTableHead}>{h}</div>)}
              {showColHeaders && <div key="sep" style={s.benchTableSeparator}></div>}
              {rows.map(r => [<div key={`lbl-${r.label}`} style={s.benchTableLabel}>{r.label}</div>, ...r.values.map((v, i) => <div key={`val-${r.label}-${i}`} style={v === null ? {
        ...s.benchTableValue,
        ...s.benchTableValueMissing
      } : s.benchTableValue}>
                    {v !== null ? v : "—"}
                  </div>)])}
            </div>
          </div>
          {legend && <div style={s.benchLegend}>
              {(Array.isArray(legend) ? legend : [legend]).map((line, i) => <div key={`legend-${i}`}>{line}</div>)}
            </div>}
        </div>;
    };
    const buildSpeedTable = measurements => {
      if (measurements.length === 0) return null;
      const {shared, differing} = partitionWorkload(measurements);
      const sharedText = formatWorkloadParts(measurements[0] && measurements[0].workload, shared);
      const colHeaders = measurements.map(m => formatWorkloadParts(m && m.workload, differing));
      const rows = SPEED_LABELS.map(tup => {
        const [key, label, unit, compute] = tup;
        const values = measurements.map(m => {
          const raw = compute ? compute(m) : m[key];
          return fmt(raw, unit);
        });
        return {
          label,
          values
        };
      });
      return {
        title: "Speed",
        sharedText,
        colHeaders,
        rows,
        colCount: measurements.length,
        legend: [`throughput per gpu = (input+output tokens)/elapsed/GPU`, `interactivity = 1000/TPOT(ms) (tokens/s/user)`]
      };
    };
    const buildAccuracyTable = accuracy => {
      if (!accuracy) return null;
      const rows = ACCURACY_LABELS.map(([key, label, unit]) => {
        const v = fmt(accuracy[key], unit);
        if (v === null) return null;
        return {
          label,
          values: [v]
        };
      }).filter(r => r !== null);
      if (rows.length === 0) return null;
      return {
        title: "Accuracy",
        sharedText: null,
        colHeaders: [],
        rows,
        colCount: 1
      };
    };
    const accuracy = effectiveAccuracy(entry, sel);
    const isEmpty = benchmarkIsEmpty(entry, accuracy);
    const measurements = !isEmpty ? normalizeSpeed(entry && entry.speed) : [];
    const accuracyTable = !isEmpty ? buildAccuracyTable(accuracy) : null;
    const speedTable = !isEmpty ? buildSpeedTable(measurements) : null;
    const hasBenchCmds = !isEmpty && buildBenchCommands(entry, sel) !== null;
    return <div style={s.benchCard}>
        <div style={s.benchHeader}>
          <div style={s.benchTitle}>Benchmark</div>
          <div style={s.benchHeaderRight}>
            {!isEmpty && entry && entry.sglang_version && <div style={s.benchVersion}>measured on sglang <code>{entry.sglang_version}</code></div>}
            {hasBenchCmds && <button style={s.iconButton} onClick={() => setModal("bench")}>⚡ Reproduce</button>}
          </div>
        </div>
        {isEmpty ? <div style={s.benchEmpty}>
            Benchmark data pending for this combination — submit yours via the Playground's Submit ↗ button.
          </div> : <>
            {accuracyTable && renderBenchTable(accuracyTable)}
            {speedTable && renderBenchTable(speedTable)}
            {entry && entry.notes && <div style={s.benchNotes}>{entry.notes}</div>}
          </>}
      </div>;
  };
  const buildBenchCommands = (entry, sel) => {
    const bc = config.benchmarkCommands;
    if (!bc) return null;
    const acc = effectiveAccuracy(entry, sel);
    const accuracy = [];
    if (bc.accuracy) {
      for (const [key, label] of ACCURACY_LABELS) {
        if (acc[key] == null) continue;
        const tmpl = bc.accuracy[key];
        const resolved = typeof tmpl === "string" ? tmpl : tmpl && tmpl[sel.variant] || null;
        if (resolved) accuracy.push({
          key,
          label,
          template: resolved
        });
      }
    }
    let speed = null;
    if (bc.speed && entry) {
      const ms = normalizeSpeed(entry.speed).filter(m => m && m.workload && m.workload.max_concurrency != null);
      const concurrencies = [...new Set(ms.map(m => m.workload.max_concurrency))].sort((a, b) => a - b);
      if (concurrencies.length) {
        speed = {
          template: bc.speed,
          concurrencies,
          workload: ms[0].workload,
          numPromptsOf: c => {
            const m = ms.find(x => x.workload.max_concurrency === c);
            if (m && m.workload.num_prompts != null) return m.workload.num_prompts;
            const tbl = bc.numPromptsByConc;
            if (tbl && tbl[c] != null) return tbl[c];
            return Math.max(c * 2, 200);
          }
        };
      }
    }
    if (accuracy.length === 0 && !speed) return null;
    return {
      accuracy,
      speed
    };
  };
  const buildHardwareGroups = () => {
    const supported = new Set(config.supportedHardware);
    const catalog = {};
    for (const [vendor, list] of Object.entries(HARDWARE_CATALOG)) catalog[vendor] = [...list];
    for (const hw of config.hardware || []) {
      const vendor = hw.vendor || "nvidia";
      const list = catalog[vendor] || (catalog[vendor] = []);
      const entry = {
        id: hw.id,
        label: hw.label,
        vram: hw.vram
      };
      const i = list.findIndex(x => x.id === hw.id);
      if (i >= 0) list[i] = entry; else list.push(entry);
    }
    const groups = [];
    for (const [vendor, list] of Object.entries(catalog)) {
      const items = list.filter(hw => supported.has(hw.id)).map(hw => ({
        id: hw.id,
        label: hw.label,
        subtitle: hw.vram
      }));
      if (items.length) groups.push({
        label: vendor.toUpperCase(),
        items
      });
    }
    if (config.groupHardware === false) {
      return [{
        label: null,
        items: groups.flatMap(group => group.items)
      }];
    }
    return groups;
  };
  const initialSelectionFromCells = () => {
    const first = (config.cells || [])[0];
    const sel = Object.fromEntries(DIMENSIONS.map(d => [d, first ? first.match[d] : ""]));
    for (const spec of overlayDimSpecs) {
      const opts = spec.options || [];
      sel[spec.id] = (spec.default ?? (opts[0] && opts[0].id)) ?? "";
    }
    if (!commandBuilder) return sel;
    return {
      ...sel,
      hw: commandBuilder.defaultSelection?.hw || config.supportedHardware?.[0] || "",
      ...commandBuilder.defaultSelection || ({})
    };
  };
  const normalizeBuilderSelection = parsed => {
    const out = {
      ...initialSelectionFromCells(),
      ...parsed
    };
    if (!(config.supportedHardware || []).includes(out.hw)) {
      out.hw = commandBuilder.defaultSelection?.hw || config.supportedHardware?.[0] || "";
    }
    for (const spec of overlayDimSpecs) {
      if (spec.kind === "number") {
        const value = Number.parseInt(out[spec.id], 10);
        out[spec.id] = Math.min(spec.max, Math.max(spec.min, Number.isFinite(value) ? value : Number(spec.default ?? spec.min)));
        continue;
      }
      const options = spec.options || [];
      if (!options.some(option => option.id === out[spec.id])) {
        out[spec.id] = (spec.default ?? options[0]?.id) ?? "";
      }
    }
    for (const [key, bounds] of Object.entries(commandBuilder.resource?.limits || ({}))) {
      const fallback = Number((commandBuilder.defaultSelection?.[key] ?? bounds.min) ?? 1);
      const value = Number.parseInt(out[key], 10);
      out[key] = Math.min(bounds.max, Math.max(bounds.min, Number.isFinite(value) ? value : fallback));
    }
    for (const key of ["tp_size", "ulysses_degree", "ring_degree"]) {
      const value = Number.parseInt(out[key], 10);
      out[key] = Number.isFinite(value) && value > 0 ? value : 1;
    }
    out.topology_mode = out.topology_mode === "manual" ? "manual" : "auto";
    return out;
  };
  const placeholderDefaults = schema => {
    const out = {};
    for (const [k, v] of Object.entries(schema || ({}))) out[k] = v.default ?? "";
    return out;
  };
  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    const check = () => {
      const html = document.documentElement;
      setIsDark(html.classList.contains("dark") || html.getAttribute("data-theme") === "dark" || html.style.colorScheme === "dark");
    };
    check();
    const observer = new MutationObserver(check);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class", "data-theme", "style"]
    });
    return () => observer.disconnect();
  }, []);
  const STORAGE_KEY = "sglang-deploy-env";
  const [env, setEnv] = useState(() => placeholderDefaults(config.placeholders));
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        setEnv({
          ...placeholderDefaults(config.placeholders),
          ...parsed
        });
      }
    } catch {}
  }, []);
  const saveEnv = next => {
    setEnv(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {}
  };
  const [sel, setSel] = useState(() => initialSelectionFromCells());
  const [selectionHydrated, setSelectionHydrated] = useState(false);
  const INTERNAL_HASH_STATE_KEY = "__sglangDeployInternalHash";
  const DEPLOYMENT_COMPONENT_ID = "deployment-configurator";
  useEffect(() => {
    const hydrate = () => {
      const raw = window.location.hash.replace(/^#/, "");
      if (!raw) return;
      const params = new URLSearchParams(raw);
      const initial = initialSelectionFromCells();
      const parsed = {
        ...initial
      };
      let touched = false;
      params.forEach((value, key) => {
        if ((key in parsed)) {
          parsed[key] = value;
          touched = true;
        }
      });
      if (!touched) return;
      setSel(commandBuilder ? normalizeBuilderSelection(parsed) : validateSelection(config.cells, parsed));
      const historyState = window.history.state;
      const isInternalHash = historyState && typeof historyState === "object" && historyState[INTERNAL_HASH_STATE_KEY] === `#${raw}`;
      if (isInternalHash) return;
      const el = document.getElementById(DEPLOYMENT_COMPONENT_ID);
      if (el) el.scrollIntoView({
        behavior: "smooth",
        block: "start"
      });
    };
    hydrate();
    setSelectionHydrated(true);
    window.addEventListener("hashchange", hydrate);
    return () => window.removeEventListener("hashchange", hydrate);
  }, []);
  useEffect(() => {
    if (!selectionHydrated) return;
    const target = "#" + new URLSearchParams(sel).toString();
    if (window.location.hash !== target) {
      const historyState = window.history.state && typeof window.history.state === "object" ? window.history.state : {};
      window.history.replaceState({
        ...historyState,
        [INTERNAL_HASH_STATE_KEY]: target
      }, "", target);
    }
    window.dispatchEvent(new CustomEvent("sglang-deploy-sel", {
      detail: sel
    }));
  }, [sel, selectionHydrated]);
  const [modal, setModal] = useState(null);
  useEffect(() => {
    if (modal === null) return;
    const onKey = e => {
      if (e.key === "Escape") setModal(null);
    };
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [modal]);
  const [copied, setCopied] = useState(false);
  const [curlCopied, setCurlCopied] = useState(false);
  const [envDraft, setEnvDraft] = useState(env);
  const [benchConc, setBenchConc] = useState(null);
  const [benchAcc, setBenchAcc] = useState(null);
  const [benchCopied, setBenchCopied] = useState(null);
  const configuredRunModes = typeof config.runModes === "function" ? config.runModes(sel) : config.runModes;
  const runModes = configuredRunModes || ["python", "docker"];
  const [runMode, setRunMode] = useState(runModes[0]);
  const [builderScope, setBuilderScope] = useState("base");
  const [builderServerSetting, setBuilderServerSetting] = useState(null);
  const [builderAdvanced, setBuilderAdvanced] = useState(false);
  const [serveExpanded, setServeExpanded] = useState(false);
  const [requestExpanded, setRequestExpanded] = useState(false);
  const [builderHeadAddress, setBuilderHeadAddress] = useState("<head-node-ip>");
  const [builderNodeRank, setBuilderNodeRank] = useState(0);
  const [blockedNote, setBlockedNote] = useState(null);
  const flashBlockedNote = (dim, reason) => {
    const note = {
      dim,
      reason
    };
    setBlockedNote(note);
    setTimeout(() => setBlockedNote(cur => cur === note ? null : cur), 4000);
  };
  useEffect(() => {
    if (builderNodeRank >= Number(sel.nodes || 1)) setBuilderNodeRank(0);
  }, [sel.nodes, builderNodeRank]);
  const hasRunMode = runModes.includes(runMode);
  const fallbackRunMode = runModes[0];
  const activeRunMode = hasRunMode ? runMode : fallbackRunMode;
  useEffect(() => {
    if (!hasRunMode) setRunMode(fallbackRunMode);
  }, [hasRunMode, fallbackRunMode]);
  useEffect(() => {
    if (modal === "env") setEnvDraft(env);
  }, [modal, env]);
  const [mambaRatio, setMambaRatio] = useState(null);
  useEffect(() => {
    const onRatio = e => setMambaRatio(e.detail && (e.detail.baseRatio || e.detail.ratio) || null);
    window.addEventListener("sglang-k3-mamba-ratio", onRatio);
    return () => window.removeEventListener("sglang-k3-mamba-ratio", onRatio);
  }, []);
  const s = makeStyles(isDark);
  const cell = commandBuilder ? commandBuilder.resolveDeployment(sel) : findCell(config.cells, sel);
  const builderMeta = cell && cell.builder || ({});
  const verifyStatus = cellVerifyStatus(cell, sel);
  const cellWithRatio = (() => {
    if (!cell || !mambaRatio) return cell;
    if (cell.flags.some(f => f.startsWith("--mamba-full-memory-ratio") || f.startsWith("--max-mamba-cache-size"))) return cell;
    const flags = [...cell.flags];
    const line = `--mamba-full-memory-ratio ${mambaRatio}`;
    const i = flags.findIndex(f => f.startsWith("--host"));
    if (i >= 0) flags.splice(i, 0, line); else flags.push(line);
    return {
      ...cell,
      flags
    };
  })();
  const commandEnv = commandBuilder ? {
    ...env,
    NODE_RANK: String(builderNodeRank),
    NODE0_IP: builderHeadAddress || "<head-node-ip>"
  } : env;
  const command = renderCommand(cellWithRatio, sel, commandEnv, activeRunMode);
  const effFlags = cell ? overlayCompose(cell.flags, sel) : [];
  const specAlgoFlag = effFlags.find(f => f.split(/[\s=]/)[0] === "--speculative-algorithm");
  const specMrrFlag = effFlags.find(f => f.split(/[\s=]/)[0] === "--max-running-requests");
  const mtpHint = !!specAlgoFlag && !specMrrFlag;
  const specPinnedHint = !!specAlgoFlag && !!specMrrFlag;
  const specMrrValue = specMrrFlag ? specMrrFlag.split(/[\s=]/).filter(Boolean)[1] || "" : "";
  const SPEC_ALGO_LABEL = {
    EAGLE: "MTP",
    EAGLE3: "MTP",
    FROZEN_KV_MTP: "MTP",
    DSPARK: "DSpark",
    DFLASH: "DFlash",
    NGRAM: "N-gram",
    STANDALONE: "standalone draft"
  };
  const specAlgoName = (() => {
    if (!specAlgoFlag) return "MTP";
    const v = specAlgoFlag.split(/[\s=]/).filter(Boolean)[1] || "";
    return SPEC_ALGO_LABEL[v.toUpperCase()] || v || "MTP";
  })();
  const renderWarn = text => {
    const out = [];
    const re = /\[([^\]]+)\]\(#([^)]+)\)/g;
    let last = 0;
    for (let m; m = re.exec(text); last = m.index + m[0].length) {
      if (m.index > last) out.push(text.slice(last, m.index));
      const anchor = m[2];
      out.push(<button key={m.index} type="button" onClick={() => {
        const el = document.getElementById(anchor);
        if (el) el.scrollIntoView({
          behavior: "smooth",
          block: "start"
        });
      }} style={{
        background: "transparent",
        border: "none",
        padding: 0,
        color: isDark ? "#FDBA74" : "#C2410C",
        cursor: "pointer",
        font: "inherit",
        fontWeight: 600,
        textDecoration: "underline",
        textUnderlineOffset: "2px"
      }}>
          {m[1]}
        </button>);
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  };
  const modelName = resolveModelName(sel);
  const curlTemplate = typeof config.curl === "function" ? config.curl(sel, cell) : config.curl;
  const curlText = interpolate(curlTemplate || "", env, modelName);
  const hwGroups = buildHardwareGroups();
  const benchEntry = benchmarks ? findBenchmark(benchmarks, sel) : null;
  const isOverlayDim = dim => overlayDimSpecs.some(d => d.id === dim);
  const findOption = (dim, value) => {
    const spec = [...matchDimSpecs, ...overlayDimSpecs].find(d => d.id === dim);
    return spec && (spec.options || []).find(o => o.id === value);
  };
  const isEnabled = (dim, value) => {
    const opt = findOption(dim, value);
    if (opt && optionDisabled(opt, sel)) return false;
    if (commandBuilder && dim === "hw") return true;
    return isOverlayDim(dim) || isOptionAvailable(config.cells || [], sel, dim, value);
  };
  const reseatHiddenPicks = next => {
    let out = next;
    for (const spec of [...matchDimSpecs, ...overlayDimSpecs]) {
      const opts = visibleOptions(spec, out).filter(o => !optionDisabled(o, out));
      if (!opts.length) continue;
      if (!opts.some(o => o.id === out[spec.id])) {
        out = {
          ...out,
          [spec.id]: opts[0].id
        };
      }
    }
    return out;
  };
  const recommendedBuilderRecipe = hw => {
    const recipes = commandBuilder.resource?.verifiedRecipes || [];
    return recipes.find(entry => entry.hw === hw && entry.default) || recipes.find(entry => entry.hw === hw);
  };
  const handleSelect = (dim, value) => {
    if (commandBuilder) {
      setSel(prev => {
        let next = {
          ...prev,
          [dim]: value
        };
        if (dim === "hw") {
          const currentRecipe = recommendedBuilderRecipe(prev.hw);
          const nextRecipe = recommendedBuilderRecipe(value);
          const resourcesFollowPlatformDefault = !!currentRecipe && Number(prev.nodes) === Number(currentRecipe.nodes) && Number(prev.gpus_per_node) === Number(currentRecipe.gpus_per_node);
          next = {
            ...next,
            nodes: resourcesFollowPlatformDefault ? nextRecipe?.nodes ?? next.nodes : next.nodes,
            gpus_per_node: resourcesFollowPlatformDefault ? nextRecipe?.gpus_per_node ?? next.gpus_per_node : next.gpus_per_node,
            topology_mode: "auto",
            tp_size: resourcesFollowPlatformDefault ? nextRecipe?.tp_size ?? 1 : next.tp_size,
            ulysses_degree: resourcesFollowPlatformDefault ? nextRecipe?.ulysses_degree ?? 1 : next.ulysses_degree,
            ring_degree: resourcesFollowPlatformDefault ? nextRecipe?.ring_degree ?? 1 : next.ring_degree,
            placement: resourcesFollowPlatformDefault ? nextRecipe?.placement || "auto" : next.placement,
            encoder: resourcesFollowPlatformDefault ? nextRecipe?.encoder || "auto" : next.encoder
          };
        }
        return reseatHiddenPicks(normalizeBuilderSelection(next));
      });
      return;
    }
    setSel(prev => reseatHiddenPicks(isOverlayDim(dim) ? {
      ...prev,
      [dim]: value
    } : snapToValidCell(config.cells, prev, dim, value)));
  };
  const commitBuilderNumber = (event, currentValue, bounds, commit) => {
    const parsed = Number(event.currentTarget.value);
    if (!Number.isInteger(parsed)) {
      event.currentTarget.value = String(currentValue);
      return;
    }
    const value = Math.min(bounds.max, Math.max(bounds.min, parsed));
    event.currentTarget.value = String(value);
    commit(value);
  };
  const renderBuilderNumberInput = ({identity, value, min, max, label, onCommit}) => <input key={identity} type="number" inputMode="numeric" min={min} max={max} step="1" defaultValue={value} aria-label={label} onFocus={event => event.currentTarget.select()} onBlur={event => commitBuilderNumber(event, value, {
    min,
    max
  }, onCommit)} onKeyDown={event => {
    if (event.key === "Enter") event.currentTarget.blur();
  }} />;
  const updateBuilderResource = (key, delta) => {
    if (!commandBuilder) return;
    const bounds = commandBuilder.resource?.limits?.[key] || ({
      min: 1,
      max: 8
    });
    setSel(prev => {
      const value = Math.min(bounds.max, Math.max(bounds.min, Number(prev[key]) + delta));
      return normalizeBuilderSelection({
        ...prev,
        [key]: value,
        topology_mode: "auto"
      });
    });
  };
  const setBuilderResource = (key, rawValue) => {
    if (!commandBuilder) return;
    const value = Number.parseInt(rawValue, 10);
    if (!Number.isFinite(value)) return;
    const bounds = commandBuilder.resource?.limits?.[key] || ({
      min: 1,
      max: 8
    });
    setSel(prev => normalizeBuilderSelection({
      ...prev,
      [key]: Math.min(bounds.max, Math.max(bounds.min, value)),
      topology_mode: "auto"
    }));
  };
  const editBuilderTopology = (key, value) => {
    if (!commandBuilder) return;
    setSel(prev => normalizeBuilderSelection({
      ...prev,
      topology_mode: "manual",
      [key]: Number.parseInt(value, 10) || 1
    }));
  };
  const handleCopy = () => {
    navigator.clipboard.writeText(command);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };
  const copyCurl = () => {
    navigator.clipboard.writeText(curlText);
    setCurlCopied(true);
    setTimeout(() => setCurlCopied(false), 1200);
  };
  const copyBench = (key, text) => {
    navigator.clipboard.writeText(text);
    setBenchCopied(key);
    setTimeout(() => setBenchCopied(null), 1200);
  };
  const placeholderGroups = (() => {
    const out = {
      command: [],
      curl: []
    };
    for (const [key, meta] of Object.entries(config.placeholders || ({}))) {
      (out[meta.target] || (out[meta.target] = [])).push({
        key,
        ...meta
      });
    }
    return out;
  })();
  const renderButton = (item, dim, selectedId) => {
    const checked = selectedId === item.id;
    const disabled = !isEnabled(dim, item.id);
    return <label key={item.id} className="sg-command-visualizer-choice" role="radio" aria-checked={checked} aria-disabled={disabled} tabIndex={disabled ? -1 : 0} style={{
      ...s.labelBase,
      ...checked ? s.checked : {},
      ...disabled ? s.disabled : {}
    }} title={disabled ? (typeof item.disableReason === "function" ? item.disableReason(sel) : item.disableReason) || "Not supported for current selection" : ""} onClick={e => {
      if (disabled) {
        e.preventDefault();
        return;
      }
      handleSelect(dim, item.id);
    }} onKeyDown={e => {
      if (disabled || e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      handleSelect(dim, item.id);
    }}>
        <input type="radio" checked={checked} disabled={disabled} readOnly style={{
      display: "none"
    }} />
        <span>{item.label}</span>
        {item.subtitle && <small style={{
      ...s.subtitle,
      color: checked ? "rgba(255,255,255,0.85)" : "inherit"
    }}>
            {item.subtitle}
          </small>}
      </label>;
  };
  const renderFlatSection = (title, options, dim, selectedId) => <div style={s.card}>
      <div style={s.title}>{title}</div>
      <div style={s.itemsGrid(options.length)}>
        {options.map(item => renderButton(item, dim, selectedId))}
      </div>
    </div>;
  const maxHwCols = Math.max(...hwGroups.map(x => x.items.length));
  if (commandBuilder) {
    const scopeLabel = {
      base: "Setup",
      serve: "Server",
      request: "Request"
    };
    const scopedDims = scope => overlayDimSpecs.filter(dim => {
      if ((dim.scope || "base") !== scope) return false;
      if (dim.kind === "number") {
        return typeof dim.showWhen !== "function" || dim.showWhen(sel);
      }
      return rowVisible(dim, sel);
    });
    const baseDims = scopedDims("base");
    const serveDims = scopedDims("serve");
    const requestDims = scopedDims("request");
    const errors = builderMeta.errors || [];
    const warnings = builderMeta.warnings || [];
    const invalid = errors.length > 0;
    const totalGpus = Number(sel.nodes) * Number(sel.gpus_per_node);
    const topology = builderMeta.topology || ({});
    const verification = builderMeta.verification || ({});
    const scopeIsVerified = scope => scopedDims(scope).every(dim => {
      const option = (dim.options || []).find(entry => entry.id === sel[dim.id]);
      if (option && optionSoft(option, sel)) return false;
      const predicate = option?.verifiedWhen ?? dim.verifiedWhen;
      return typeof predicate === "function" ? !!predicate(sel) : predicate !== false;
    });
    const serveStatus = invalid ? "error" : scopeIsVerified("serve") ? verification.serve || verifyStatus : "unverified";
    const requestStatus = invalid ? "error" : scopeIsVerified("request") ? verification.request || verifyStatus : "unverified";
    const statusText = status => ({
      verified: "Verified",
      unverified: "Unverified",
      "in-progress": "Verification in progress",
      error: "Invalid configuration"
    })[status] || "Unverified";
    const activeServerSetting = serveDims.find(dim => dim.id === builderServerSetting) || serveDims[0];
    const selectedOption = dim => (dim.options || []).find(option => option.id === sel[dim.id]);
    const effectiveSetting = dim => builderMeta.resolvedSettings?.[dim.id] || selectedOption(dim)?.label || sel[dim.id] || "—";
    const recommendedRecipe = recommendedBuilderRecipe(sel.hw);
    const recommendedInUse = !!recommendedRecipe && Number(sel.nodes) === recommendedRecipe.nodes && Number(sel.gpus_per_node) === recommendedRecipe.gpus_per_node && sel.topology_mode === "auto" && serveDims.every(dim => {
      const expected = recommendedRecipe[dim.id] ?? dim.default;
      if (dim.id === "attention") return sel.attention === "platform";
      return expected === undefined || sel[dim.id] === expected || ["placement", "encoder"].includes(dim.id) && sel[dim.id] === "auto";
    });
    const restoreRecommendedRecipe = () => {
      if (!recommendedRecipe) return;
      setSel(prev => reseatHiddenPicks(normalizeBuilderSelection({
        ...prev,
        ...Object.fromEntries(serveDims.map(dim => [dim.id, (recommendedRecipe[dim.id] ?? dim.default) ?? dim.options?.[0]?.id])),
        nodes: recommendedRecipe.nodes,
        gpus_per_node: recommendedRecipe.gpus_per_node,
        topology_mode: "auto",
        tp_size: recommendedRecipe.tp_size,
        ulysses_degree: recommendedRecipe.ulysses_degree,
        ring_degree: recommendedRecipe.ring_degree,
        placement: recommendedRecipe.placement || "auto",
        attention: "platform",
        precision: "native",
        encoder: recommendedRecipe.encoder || "auto",
        execution: recommendedRecipe.execution || "eager"
      })));
    };
    const renderBuilderChoice = (item, dim) => {
      const checked = sel[dim.id] === item.id;
      const disabled = !isEnabled(dim.id, item.id);
      const soft = !disabled && optionSoft(item, sel);
      const reason = disabled ? item.disableReason || "Not available for this configuration" : soft ? item.softReason || "Runs, but this combination is not a verified recipe yet." : "";
      return <button key={item.id} type="button" className="sgd-builder-choice" data-selected={checked ? "true" : "false"} data-blocked={disabled ? "true" : undefined} data-soft={soft ? "true" : undefined} aria-disabled={disabled} aria-pressed={checked} title={reason} onClick={() => {
        if (disabled) {
          flashBlockedNote(dim.id, reason);
          return;
        }
        handleSelect(dim.id, item.id);
      }}>
          <span className="sgd-builder-choice-dot" aria-hidden="true" />
          <span>{item.label}</span>
          {item.subtitle && <small>{item.subtitle}</small>}
        </button>;
    };
    const renderBuilderDimension = dim => <section className="sgd-builder-section" key={dim.id}>
        <div className="sgd-builder-section-heading">
          <span>{dim.title}</span>
          {dim.description && <small>{dim.description}</small>}
        </div>
        <div className="sgd-builder-choice-grid" data-density={(dim.options || []).length > 5 ? "compact" : "normal"}>
          {visibleOptions(dim, sel).map(option => renderBuilderChoice(option, dim))}
        </div>
        {blockedNote && blockedNote.dim === dim.id && <p className="sgd-builder-blocked-note" role="status">{blockedNote.reason}</p>}
      </section>;
    const renderStepper = (key, label, detail) => {
      const bounds = commandBuilder.resource?.limits?.[key] || ({
        min: 1,
        max: 8
      });
      return <div className="sgd-builder-stepper-field">
          <div>
            <span>{label}</span>
            {detail && <small>{detail}</small>}
          </div>
          <div className="sgd-builder-stepper" aria-label={label}>
            <button type="button" aria-label={`Decrease ${label}`} disabled={Number(sel[key]) <= bounds.min} onClick={() => updateBuilderResource(key, -1)}>−</button>
            {renderBuilderNumberInput({
        identity: `${key}-${sel[key]}`,
        value: sel[key],
        min: bounds.min,
        max: bounds.max,
        label,
        onCommit: value => setBuilderResource(key, value)
      })}
            <button type="button" aria-label={`Increase ${label}`} disabled={Number(sel[key]) >= bounds.max} onClick={() => updateBuilderResource(key, 1)}>+</button>
          </div>
        </div>;
    };
    const renderBaseScope = () => <div className="sgd-builder-scope-panel" data-scope="base">
        {recommendedRecipe && <section className="sgd-builder-recipe">
            <div>
              {}
              <span>{recommendedRecipe.unverified ? "Derived recipe" : "Verified recipe"} · {sel.hw.toUpperCase()}</span>
              <strong>
                {[`${recommendedRecipe.nodes * recommendedRecipe.gpus_per_node} GPUs`, recommendedRecipe.tp_size > 1 && `TP ${recommendedRecipe.tp_size}`, `Ulysses ${recommendedRecipe.ulysses_degree}`, recommendedRecipe.ring_degree > 1 && `Ring ${recommendedRecipe.ring_degree}`, ({
      resident: "Resident",
      fsdp: "FSDP",
      offload: "Layerwise offload"
    })[recommendedRecipe.placement]].filter(Boolean).join(" · ")}
              </strong>
            </div>
            <div>
              {renderStatus(recommendedRecipe.unverified ? "unverified" : "verified")}
              {recommendedInUse ? <small>In use</small> : <button type="button" className="sgd-builder-text-action" onClick={restoreRecommendedRecipe}>{recommendedRecipe.unverified ? "Use derived recipe" : "Use verified recipe"}</button>}
            </div>
          </section>}
        <section className="sgd-builder-section">
          <div className="sgd-builder-section-heading"><span>Hardware</span></div>
          <div className="sgd-builder-hardware-grid">
            {hwGroups.flatMap(group => group.items).map(item => {
      const selected = sel.hw === item.id;
      return <button key={item.id} type="button" className="sgd-builder-hardware" data-selected={selected ? "true" : "false"} aria-pressed={selected} onClick={() => handleSelect("hw", item.id)}>
                  <span className="sgd-builder-choice-dot" aria-hidden="true" />
                  <strong>{item.label}</strong>
                  <small>{item.subtitle}</small>
                </button>;
    })}
          </div>
        </section>

        {}
        <section className="sgd-builder-section">
          <div className="sgd-builder-section-heading">
            <span>Resources</span>
            <small>{builderMeta.topologySummary || "No valid topology"}</small>
          </div>
          <div className="sgd-builder-resource-grid">
            {renderStepper("nodes", "Nodes")}
            {renderStepper("gpus_per_node", "GPUs / node")}
          </div>
          {Number(sel.nodes) > 1 && <p className="sgd-builder-resource-summary">
              {sel.nodes} nodes × {sel.gpus_per_node} {sel.hw.toUpperCase()} = {totalGpus} GPUs
            </p>}
          <button type="button" className="sgd-builder-text-action sgd-builder-topology-toggle" aria-expanded={builderAdvanced} onClick={() => setBuilderAdvanced(open => !open)}>
            Advanced topology <span aria-hidden="true">{builderAdvanced ? "↗" : "↘"}</span>
          </button>
          {builderAdvanced && <div className="sgd-builder-advanced">
              <p>Auto uses an exact verified recipe when one exists; manual values are allowed when the model constraints remain valid.</p>
              <div className="sgd-builder-topology-inputs">
                {[["tp_size", "Tensor parallel", [1, 2, 4, 8]], ["ulysses_degree", "Ulysses", [1, 2, 4, 8, 16]], ["ring_degree", "Ring", [1, 2, 4, 8]]].map(([key, label, values]) => <label key={key}>
                    <span>{label}</span>
                    <select value={sel.topology_mode === "manual" ? sel[key] : topology[key] || 1} onChange={event => editBuilderTopology(key, event.target.value)}>
                      {values.map(value => <option value={value} key={value}>{value}</option>)}
                    </select>
                  </label>)}
              </div>
              <button type="button" className="sgd-builder-text-action" disabled={sel.topology_mode === "auto"} onClick={() => setSel(prev => normalizeBuilderSelection({
      ...prev,
      topology_mode: "auto"
    }))}>Use automatic topology</button>
            </div>}
          {(errors.length > 0 || warnings.length > 0) && <div className="sgd-builder-messages" data-state={errors.length ? "error" : "warning"}>
              {(errors.length ? errors : warnings).map((message, index) => <p key={index}>{message}</p>)}
            </div>}
        </section>

        {baseDims.map(renderBuilderDimension)}
      </div>;
    const renderSettingEditor = (dim, className = "", direct = false) => {
      if (!dim) return null;
      const options = visibleOptions(dim, sel);
      const currentOption = selectedOption(dim);
      return <section className={["sgd-builder-context", className].filter(Boolean).join(" ")} aria-live={direct ? undefined : "polite"}>
          <div className="sgd-builder-context-heading">
            <div>
              <span>{direct ? dim.title : `${dim.title} options`}</span>
              {dim.description && <p>{dim.description}</p>}
            </div>
            {dim.quality && <small>{dim.quality}</small>}
          </div>
          {dim.kind === "number" ? <div className="sgd-builder-request-stepper">
              <button type="button" aria-label={`Decrease ${dim.title}`} disabled={Number(sel[dim.id]) <= dim.min} onClick={() => setSel(prev => ({
        ...prev,
        [dim.id]: Math.max(dim.min, Number(prev[dim.id]) - 1)
      }))}>−</button>
              {renderBuilderNumberInput({
        identity: `${dim.id}-${sel[dim.id]}`,
        value: sel[dim.id],
        min: dim.min,
        max: dim.max,
        label: dim.title,
        onCommit: value => setSel(prev => ({
          ...prev,
          [dim.id]: value
        }))
      })}
              <button type="button" aria-label={`Increase ${dim.title}`} disabled={Number(sel[dim.id]) >= dim.max} onClick={() => setSel(prev => ({
        ...prev,
        [dim.id]: Math.min(dim.max, Number(prev[dim.id]) + 1)
      }))}>+</button>
              <span>{dim.unit || "outputs"}</span>
            </div> : <div className="sgd-builder-context-options">
              {options.map(option => renderBuilderChoice(option, dim))}
            </div>}
          {blockedNote && blockedNote.dim === dim.id && <p className="sgd-builder-blocked-note" role="status">{blockedNote.reason}</p>}
          {(currentOption?.description || dim.learnMore) && <div className="sgd-builder-context-note">
              {currentOption?.description && <p>{currentOption.description}</p>}
              {}
              {dim.learnMore && <a href={dim.learnMore}>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><line x1="4" y1="6" x2="20" y2="6" /><line x1="4" y1="12" x2="16" y2="12" /><line x1="4" y1="18" x2="11" y2="18" /></svg>
                  Learn more
                </a>}
              {dim.docsHref && <a href={dim.docsHref}>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" /></svg>
                  SGLang docs
                </a>}
            </div>}
        </section>;
    };
    const renderServerScope = () => <div className="sgd-builder-scope-panel" data-scope="serve">
        <div className="sgd-builder-setting-layout">
          <div className="sgd-builder-setting-list">
            {serveDims.map(dim => {
      const isActive = dim.id === activeServerSetting?.id;
      const option = selectedOption(dim);
      const recommended = typeof option?.recommendedWhen === "function" ? option.recommendedWhen(sel) : !!option?.recommended;
      return <div className="sgd-builder-setting-item" key={dim.id}>
                  <button type="button" className="sgd-builder-setting-row" data-active={isActive ? "true" : "false"} aria-expanded={isActive} onClick={() => setBuilderServerSetting(dim.id)}>
                    <span>{dim.title}</span>
                    <strong>{effectiveSetting(dim)}</strong>
                    {recommended && <small>Recommended</small>}
                    <span aria-hidden="true">{isActive ? "⌄" : "›"}</span>
                  </button>
                  {isActive && renderSettingEditor(dim, "sgd-builder-context--inline")}
                </div>;
    })}
          </div>
          {renderSettingEditor(activeServerSetting, "sgd-builder-context--rail")}
        </div>
      </div>;
    const renderRequestScope = () => <div className="sgd-builder-scope-panel sgd-builder-request-direct" data-scope="request">
        {requestDims.map(dim => <div className="sgd-builder-request-setting" key={dim.id}>
            {renderSettingEditor(dim, "", true)}
          </div>)}
      </div>;
    const renderScopeControls = () => {
      if (builderScope === "base") return renderBaseScope();
      if (builderScope === "serve") return renderServerScope();
      return renderRequestScope();
    };
    const renderStatus = status => <span className="sgd-builder-status" data-status={status}>
        <span aria-hidden="true" />{statusText(status)}
      </span>;
    const renderOutputCard = type => {
      const serve = type === "serve";
      const text = serve ? command : curlText;
      const canExpand = text.split("\n").length > 9;
      const expanded = serve ? serveExpanded : requestExpanded;
      const setExpanded = serve ? setServeExpanded : setRequestExpanded;
      const status = serve ? serveStatus : requestStatus;
      const emphasized = builderScope === "base" || builderScope === type;
      return <section className="sgd-builder-output" data-output={type} data-emphasized={emphasized ? "true" : "false"}>
          <header>
            <div className="sgd-builder-output-index">{serve ? "1" : "2"}</div>
            <div className="sgd-builder-output-title">
              <strong>{serve ? "Serve" : "Request"}</strong>
              <span>
                {serve ? `${sel.hw.toUpperCase()} · ${activeRunMode === "docker" ? "Docker" : "Python"}` : "cURL"}
              </span>
            </div>
            {renderStatus(status)}
          </header>
          {serve && runModes.length > 1 && <div className="sgd-builder-output-tabs" role="tablist" aria-label="Serve command format">
              {runModes.map(mode => <button type="button" role="tab" aria-selected={activeRunMode === mode} data-selected={activeRunMode === mode ? "true" : "false"} key={mode} onClick={() => setRunMode(mode)}>{mode === "docker" ? "Docker" : "Python"}</button>)}
            </div>}
          {serve && Number(sel.nodes) > 1 && <div className="sgd-builder-node-fields">
              <label>
                <span>Head address</span>
                <input value={builderHeadAddress} onChange={event => setBuilderHeadAddress(event.target.value)} />
              </label>
              <label>
                <span>Node rank</span>
                {renderBuilderNumberInput({
        identity: `node-rank-${builderNodeRank}-${sel.nodes}`,
        value: builderNodeRank,
        min: 0,
        max: Number(sel.nodes) - 1,
        label: "Node rank",
        onCommit: setBuilderNodeRank
      })}
              </label>
            </div>}
          <div className="sgd-builder-code">
            <pre className={expanded ? "is-expanded" : ""}><code>{text}</code></pre>
            <button type="button" className="sgd-builder-copy" disabled={invalid} aria-label={(serve ? copied : curlCopied) ? "Copied" : "Copy command"} data-copied={(serve ? copied : curlCopied) ? "true" : undefined} onClick={serve ? handleCopy : copyCurl}>
              <svg className="sgd-builder-copy-glyph" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect x="9" y="9" width="12" height="12" rx="2.5" /><path d="M15 5v-.25A2.75 2.75 0 0 0 12.25 2h-7.5A2.75 2.75 0 0 0 2 4.75v7.5A2.75 2.75 0 0 0 4.75 15H5" /></svg>
              <svg className="sgd-builder-copy-check" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5" /></svg>
            </button>
          </div>
          {invalid && <div className="sgd-builder-output-error">{errors[0]}</div>}
          <footer>
            {canExpand && <button type="button" className="sgd-builder-text-action" onClick={() => setExpanded(!expanded)}>
                {expanded ? "Collapse" : "Expand"}
              </button>}
            <div>
              <button type="button" className="sgd-builder-text-action" onClick={() => setModal("env")}>Variables</button>
            </div>
          </footer>
        </section>;
    };
    return <section id={DEPLOYMENT_COMPONENT_ID} className="not-prose sg-command-visualizer sgd-command-builder" style={{
      scrollMarginTop: "104px"
    }} aria-label={`${config.modelName} command builder`}>
        <nav className="sgd-builder-scope-tabs" role="tablist" aria-label="Command builder scope">
          {["base", "serve", "request"].map(scope => <button type="button" role="tab" key={scope} aria-label={scopeLabel[scope]} aria-selected={builderScope === scope} aria-controls={`${DEPLOYMENT_COMPONENT_ID}-controls`} data-active={builderScope === scope ? "true" : "false"} onClick={() => setBuilderScope(scope)}>
              {scopeLabel[scope]}
            </button>)}
        </nav>

        <div className="sgd-builder-main" data-scope={builderScope}>
          <div id={`${DEPLOYMENT_COMPONENT_ID}-controls`} className="sgd-builder-controls" role="tabpanel" aria-label={`${scopeLabel[builderScope]} settings`}>
            {renderScopeControls()}
          </div>
          <div className="sgd-builder-output-rail">
            {builderScope !== "request" && renderOutputCard("serve")}
            {builderScope !== "serve" && renderOutputCard("request")}
          </div>
        </div>

        {modal === "env" && <div style={s.modalBackdrop} onClick={() => setModal(null)}>
            <div style={s.modalBox} onClick={event => event.stopPropagation()}>
              <div style={s.modalHeader}>
                <div style={s.modalTitle}>Command variables</div>
                <button style={s.modalCloseBtn} onClick={() => setModal(null)} aria-label="Close">×</button>
              </div>
              {["command", "curl"].map(target => placeholderGroups[target].length > 0 && <div key={target}>
                  <div style={s.sectionHeading}>{target === "command" ? "Serve" : "Request"}</div>
                  {placeholderGroups[target].map(({key, label}) => <div key={key} style={s.formField}>
                      <label style={s.formLabel}>{label}</label>
                      <input style={s.formInput} value={envDraft[key] ?? ""} onChange={event => setEnvDraft({
      ...envDraft,
      [key]: event.target.value
    })} />
                    </div>)}
                </div>)}
              <div style={{
      display: "flex",
      justifyContent: "flex-end",
      gap: 8,
      marginTop: 16
    }}>
                <button style={{
      ...s.iconButton,
      padding: "6px 14px"
    }} onClick={() => setModal(null)}>Cancel</button>
                <button style={s.primaryBtn} onClick={() => {
      saveEnv(envDraft);
      setModal(null);
    }}>Save</button>
              </div>
            </div>
          </div>}
      </section>;
  }
  return <div id={DEPLOYMENT_COMPONENT_ID} style={{
    ...s.container,
    scrollMarginTop: "104px"
  }} className="not-prose sg-command-visualizer">
      {}
      <div style={s.cardColumn}>
        <div style={{
    ...s.title,
    marginBottom: "2px"
  }}>Hardware Platform</div>
        {hwGroups.map(g => <div key={g.label || "hardware"} style={s.vendorRow}>
            {g.label && <div style={s.vendorLabel}>{g.label}</div>}
            <div style={s.itemsGrid(maxHwCols)}>
              {g.items.map(item => renderButton(item, "hw", sel.hw))}
              {Array.from({
    length: maxHwCols - g.items.length
  }).map((_, i) => <div key={`pad-${i}`} />)}
            </div>
          </div>)}
      </div>

      {matchDimSpecs.filter(d => rowVisible(d, sel)).map(d => <div key={d.id}>
            {renderFlatSection(d.title, visibleOptions(d, sel), d.id, sel[d.id])}
          </div>)}
      {overlayDimSpecs.filter(d => rowVisible(d, sel)).map(d => <div key={d.id}>
            {renderFlatSection(d.title, visibleOptions(d, sel), d.id, sel[d.id])}
          </div>)}

      {}
      <div style={s.card}>
        <div style={s.title}>Command:</div>
        <div style={s.commandWrap}>
          {cell && cell.redirect ? cell.warn && <div style={s.mtpWarn}>⚠️ {renderWarn(cell.warn)}</div> : <>
            <div style={s.commandHeader}>
              <div style={s.headerLeft}>
                <div style={s.badge(verifyStatus)}>
                  <span style={s.badgeDot(verifyStatus)} />
                  {VERIFY_LABEL[verifyStatus]}
                </div>
                <div style={s.runModeWrap} role="tablist" aria-label="Output format">
                  {runModes.map((mode, index) => <span key={mode} className="sg-command-visualizer-tab" style={{
    ...index === runModes.length - 1 ? s.runModeChipLast(activeRunMode === mode) : s.runModeChip(activeRunMode === mode),
    ...runModes.length === 1 ? {
      borderRadius: 7
    } : {}
  }} onClick={() => setRunMode(mode)} onKeyDown={e => {
    if (e.key !== "Enter" && e.key !== " ") return;
    e.preventDefault();
    setRunMode(mode);
  }} role="tab" tabIndex={0} aria-selected={activeRunMode === mode}>
                      {mode === "docker" ? "Docker" : "Python"}
                    </span>)}
                </div>
              </div>
              <div style={s.iconRow}>
                <button style={s.iconButton} onClick={handleCopy}>
                  {copied ? "✓ Copied" : "⧉ Copy"}
                </button>
                <button style={s.iconButton} onClick={() => setModal("curl")}>$ cURL</button>
                <button style={s.iconButton} onClick={() => setModal("env")}>⚙ Env</button>
              </div>
            </div>
            <pre style={s.commandPre}>{command}</pre>
            {cell && cell.warn && <div style={s.mtpWarn}>⚠️ {renderWarn(cell.warn)}</div>}
            {mtpHint && <div style={s.mtpWarn}>
                ⚠️ Speculative decoding ({specAlgoName}) is on — SGLang resets <code>--max-running-requests</code> to <strong>48</strong> when it isn't set. Add <code>--max-running-requests &lt;N&gt;</code> sized for your target concurrency.
              </div>}
            {specPinnedHint && <div style={s.mtpWarn}>
                ℹ️ Speculative decoding ({specAlgoName}) is on and this recipe pins <code>--max-running-requests</code> to <strong>{specMrrValue}</strong>. Adjust it to match your target concurrency — if you remove the flag, SGLang falls back to <strong>48</strong>.
              </div>}
          </>}
        </div>
      </div>

      {}
      {benchmarks && cell && renderBenchmarkCard(benchEntry)}

      {}
      {config.showPlaygroundLink !== false && <div style={{
    padding: "6px 12px",
    fontSize: "12px",
    color: isDark ? "#9ca3af" : "#6b7280",
    display: "flex",
    alignItems: "center",
    gap: "6px"
  }}>
          <span>Need to go beyond the verified matrix?</span>
          <button type="button" onClick={() => {
    const el = document.getElementById("playground");
    if (el) el.scrollIntoView({
      behavior: "smooth",
      block: "start"
    });
  }} style={{
    background: "transparent",
    border: "none",
    padding: 0,
    color: isDark ? "#FDBA74" : "#C2410C",
    cursor: "pointer",
    fontSize: "12px",
    fontWeight: 600,
    textDecoration: "underline",
    textUnderlineOffset: "2px"
  }}>
            Open the Playground →
          </button>
        </div>}

      {}
      {modal === "curl" && <div style={s.modalBackdrop} onClick={() => setModal(null)}>
          <div style={s.modalBox} onClick={e => e.stopPropagation()}>
            <div style={s.modalHeader}>
              <div style={s.modalTitle}>cURL example</div>
              <button style={s.modalCloseBtn} onClick={() => setModal(null)} aria-label="Close">×</button>
            </div>
            <div style={s.commandWrap}>
              <div style={s.commandHeader}>
                <div style={{
    fontSize: 11,
    opacity: 0.7
  }}>
                  Model: <code>{modelName || "(unresolved)"}</code>
                </div>
                <button style={s.iconButton} onClick={copyCurl}>
                  {curlCopied ? "✓ Copied" : "⧉ Copy"}
                </button>
              </div>
              <pre style={s.commandPre}>{curlText}</pre>
            </div>
            <p style={{
    fontSize: 11,
    opacity: 0.7,
    marginTop: 8
  }}>
              Edit <code>CURL_HOST</code> / <code>CURL_PORT</code> in the Env panel.
            </p>
          </div>
        </div>}

      {}
      {modal === "env" && <div style={s.modalBackdrop} onClick={() => setModal(null)}>
          <div style={s.modalBox} onClick={e => e.stopPropagation()}>
            <div style={s.modalHeader}>
              <div style={s.modalTitle}>Env / placeholder values</div>
              <button style={s.modalCloseBtn} onClick={() => setModal(null)} aria-label="Close">×</button>
            </div>
            {placeholderGroups.curl.length > 0 && <div>
                <div style={s.sectionHeading}>cURL placeholders</div>
                {placeholderGroups.curl.map(({key, label}) => <div key={key} style={s.formField}>
                    <label style={s.formLabel}>
                      {label} <code style={{
    opacity: 0.6
  }}>{`{{${key}}}`}</code>
                    </label>
                    <input style={s.formInput} value={envDraft[key] ?? ""} onChange={e => setEnvDraft({
    ...envDraft,
    [key]: e.target.value
  })} />
                  </div>)}
              </div>}
            {placeholderGroups.command.length > 0 && <div>
                <div style={s.sectionHeading}>Command placeholders</div>
                {placeholderGroups.command.map(({key, label}) => <div key={key} style={s.formField}>
                    <label style={s.formLabel}>
                      {label} <code style={{
    opacity: 0.6
  }}>{`{{${key}}}`}</code>
                    </label>
                    <input style={s.formInput} value={envDraft[key] ?? ""} onChange={e => setEnvDraft({
    ...envDraft,
    [key]: e.target.value
  })} />
                  </div>)}
              </div>}
            <div style={{
    display: "flex",
    justifyContent: "flex-end",
    gap: 8,
    marginTop: 16
  }}>
              <button style={{
    ...s.iconButton,
    padding: "6px 14px"
  }} onClick={() => setModal(null)}>Cancel</button>
              <button style={s.primaryBtn} onClick={() => {
    saveEnv(envDraft);
    setModal(null);
  }}>Save</button>
            </div>
            <p style={{
    fontSize: 11,
    opacity: 0.7,
    marginTop: 10
  }}>
              Values persist in localStorage and are reused the next time you visit any cookbook.
            </p>
          </div>
        </div>}

      {}
      {modal === "bench" && benchEntry && (() => {
    const bc = buildBenchCommands(benchEntry, sel);
    if (!bc) return null;
    const selSummary = [sel.hw && sel.hw.toUpperCase(), sel.variant, sel.quant && sel.quant.toUpperCase(), sel.strategy, sel.kvDsaPair, sel.nodes].filter(part => part !== undefined && part !== null && part !== "").join(" · ");
    let selConc = null;
    let speedCmd = null;
    if (bc.speed) {
      selConc = bc.speed.concurrencies.includes(benchConc) ? benchConc : bc.speed.concurrencies[0];
      const w = bc.speed.workload;
      speedCmd = interpolate(bc.speed.template, {
        ...env,
        DATASET: w.dataset,
        ISL: w.isl,
        OSL: w.osl,
        MAX_CONCURRENCY: selConc,
        NUM_PROMPTS: bc.speed.numPromptsOf(selConc)
      }, modelName);
    }
    let selAcc = null;
    let accCmd = null;
    if (bc.accuracy.length > 0) {
      selAcc = bc.accuracy.find(a => a.key === benchAcc) || bc.accuracy[0];
      accCmd = interpolate(selAcc.template, env, modelName);
    }
    return <div style={s.modalBackdrop} onClick={() => setModal(null)}>
            <div style={s.modalBox} onClick={e => e.stopPropagation()}>
              <div style={s.modalHeader}>
                <div style={s.modalTitle}>Benchmark commands</div>
                <button style={s.modalCloseBtn} onClick={() => setModal(null)} aria-label="Close">×</button>
              </div>
              <p style={{
      fontSize: 11,
      opacity: 0.7,
      margin: "0 0 12px"
    }}>
                For <code>{selSummary}</code>. Start the server with the Deploy command above, then run these against it.
              </p>

              {selAcc && <div>
                  <div style={s.sectionHeading}>Accuracy</div>
                  {bc.accuracy.length > 1 && <div style={s.benchChipRow}>
                      <span style={{
      fontSize: 11,
      opacity: 0.7
    }}>benchmark:</span>
                      {bc.accuracy.map(a => <button key={a.key} style={{
      ...s.benchChip,
      ...a.key === selAcc.key ? s.benchChipActive : {}
    }} onClick={() => setBenchAcc(a.key)}>
                          {a.label}
                        </button>)}
                    </div>}
                  <div style={{
      ...s.commandWrap,
      marginBottom: 6
    }}>
                    <div style={s.commandHeader}>
                      <div style={{
      fontSize: 11,
      opacity: 0.7
    }}>{selAcc.label}</div>
                      <button style={s.iconButton} onClick={() => copyBench("acc", accCmd)}>
                        {benchCopied === "acc" ? "✓ Copied" : "⧉ Copy"}
                      </button>
                    </div>
                    <pre style={s.commandPre}>{accCmd}</pre>
                  </div>
                  {bc.accuracy.length > 1 && <p style={{
      fontSize: 11,
      opacity: 0.7,
      margin: "0 0 4px"
    }}>
                      Switch the benchmark chip to see each eval's command.
                    </p>}
                </div>}

              {bc.speed && <div>
                  <div style={s.sectionHeading}>Speed</div>
                  {bc.speed.concurrencies.length > 1 && <div style={s.benchChipRow}>
                      <span style={{
      fontSize: 11,
      opacity: 0.7
    }}>max-concurrency:</span>
                      {bc.speed.concurrencies.map(c => <button key={c} style={{
      ...s.benchChip,
      ...c === selConc ? s.benchChipActive : {}
    }} onClick={() => setBenchConc(c)}>
                          {c}
                        </button>)}
                    </div>}
                  <div style={{
      ...s.commandWrap,
      marginBottom: 6
    }}>
                    <div style={s.commandHeader}>
                      <div style={{
      fontSize: 11,
      opacity: 0.7
    }}>max-concurrency = {selConc}</div>
                      <button style={s.iconButton} onClick={() => copyBench("speed", speedCmd)}>
                        {benchCopied === "speed" ? "✓ Copied" : "⧉ Copy"}
                      </button>
                    </div>
                    <pre style={s.commandPre}>{speedCmd}</pre>
                  </div>
                  <p style={{
      fontSize: 11,
      opacity: 0.7,
      margin: "0 0 4px"
    }}>
                    One command — switch the concurrency chip (or edit <code>--max-concurrency</code>) to reproduce each Speed column.
                  </p>
                </div>}

              <p style={{
      fontSize: 11,
      opacity: 0.7,
      marginTop: 12
    }}>
                Edit <code>CURL_HOST</code> / <code>CURL_PORT</code> in the Env panel.
              </p>
            </div>
          </div>;
  })()}
    </div>;
};

## Deployment

<a id="install" />

<Accordion title="Install SGLang">
  For all methods and hardware platforms, see the [official SGLang installation guide](../../../docs/get-started/install). The two paths below match the **Python / Docker** toggle in the command panel.

  <Tabs>
    <Tab title="Python (pip / uv)">
      ```bash Command theme={null}
      pip install --upgrade pip
      pip install uv

      uv pip install --prerelease=allow sglang
      ```

      Then run the **Python** output of the command panel below in that environment.
    </Tab>

    <Tab title="Docker">
      ```bash Command theme={null}
      docker pull lmsysorg/sglang:latest
      ```

      For how to launch the image, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). Substitute the inner `sglang serve ...` with what the command generator below produces.
    </Tab>
  </Tabs>
</Accordion>

Pick your card + checkpoint precision to generate the launch command. The model runs single-GPU on every supported card — H200, RTX PRO 6000, RTX 5090 and DGX Spark — and ships one operating point.

<Note>
  `--mamba-full-memory-ratio` is the one sizing flag that matters for throughput
  on hybrid GDN models: the default (0.9) over-provisions the KV pool and silently
  clamps concurrency. Set your average request length in the
  [Mamba ratio calculator](#mamba-ratio-calculator) below; everything else follows
  the panels, and the computed value is pinned into the command.
</Note>

<Deployment config={config} />

<Note>
  Every cell above — RTX 5090, RTX PRO 6000 and DGX Spark, across all five
  checkpoints and every Speculative Decoding / Serving Strategy / SSM dtype
  combination — is measured on **v0.5.19**. That is 202 cells, each one served
  and scored on the full 1319-question GSM8K (93.18-95.15%). The serving
  envelope behind the pins is ISL 8192 / OSL 1024 at concurrency 1; throughput
  and acceptance-length numbers were not re-taken in that sweep.
</Note>

### Mamba ratio calculator

<Qwen38MambaRatioCalculator />

<Accordion title="How --mamba-full-memory-ratio is calculated">
  Hybrid GDN models split post-weight memory into a worst-case-reserved **GDN
  state pool** (sets the concurrency ceiling) and a paged **attention KV pool**,
  divided by `--mamba-full-memory-ratio`. Every parameter below except `L` and the
  target concurrency is read live from the Deploy panel and Playground selection;
  the balanced value is the per-request cost ratio:

  ```text Formula theme={null}
  ratio = (S + D) x state_bytes / (L x kv_bytes_per_token)
  ```

  * `S` — state slots per running request: `extra_buffer=5` (default),
    `extra_buffer_lazy=4`, `no_buffer=3`, disabled radix cache `=1`. For the two
    `extra_buffer` strategies, `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` frees one
    slot, and `extra_buffer` frees one more with the overlap scheduler off; the
    calculator reads both knobs.
  * `D` — verify intermediate states under speculative decoding:
    `--speculative-num-draft-tokens` for EAGLE/MTP (4 at the recommended 3/1/4)
    and for DFLASH (8, DFlash2's block size);
    `--speculative-dspark-block-size + 1` for DSPARK, where the block size falls
    back to the draft checkpoint's `block_size` when the flag is omitted (7 for
    `RadixArk/Qwen3.8-27B-DSpark`, so `D = 8`); 0 with speculation off or with
    `--enable-linear-replayssm-spec`, which keeps the verify intermediates on a
    fixed ring instead of per-request slots.
  * `state_bytes` — one state slot, from the fixed geometry
    (48 GDN layers x 48 heads x 128 x 128 at `--mamba-ssm-dtype`, plus bf16 conv
    state): 153.9 MB at fp32, 78.4 MB at bf16.
  * `kv_bytes_per_token` — 16 attention layers x GQA 4 x 256 x K+V:
    32.8 KB at fp8, 65.5 KB at bf16.
  * `L` — average total request length in tokens: input + output.

  `--max-mamba-cache-size = target_concurrency x S` is the equivalent explicit
  pin and overrides the ratio; the calculator emits it alongside. `D` is not a
  term here: the engine divides the state pool by `S` alone and sizes the
  speculative verify buffer separately, so folding `D` into the pin would
  over-provision the pool. After boot, verify with the `max_running_requests`
  line in the server log — it should not be capped below your target concurrency.
</Accordion>

## Playground

The Playground is where you experiment with **SGLang features beyond the recipes above**. The Deploy panel emits this model's documented launch recipes; the Playground lets you turn on additional knobs on top of whichever cell the Deploy panel is currently showing.

<Playground config={config} />

## 1. Model Introduction

**Qwen3.8-27B** is a dense hybrid Gated Delta Networks (GDN) **vision-language**
model: a 27B causal language model paired with a vision encoder, with native
image and video understanding alongside text. SGLang serves it through the
Qwen3-VL path, so the vision tower is live on the recipes below.

The language model is 64 layers, laid out as 16 repeats of *3 × (Gated DeltaNet
→ FFN)* followed by *1 × (Gated Attention → FFN)* — 48 linear-attention layers
to 16 full-attention ones. Gated DeltaNet runs 48 value heads and 16 QK heads at
head\_dim 128; Gated Attention is GQA 24/4 at head\_dim 256 with a 64-dim rotary
slice. Hidden size is 5120 over a 17,408-dim FFN, and the checkpoint ships an
MTP head trained with multiple steps. Context is 262,144 tokens natively,
extensible to 1,000,000. The serving-relevant architecture is identical to
Qwen3.6-27B.

Thinking mode is on by default and can be disabled per request; reasoning depth
is tunable with `reasoning_effort`, and `preserve_thinking` retains reasoning
context from earlier messages.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "38%"}} />

    <col style={{width: "30%"}} />

    <col style={{width: "32%"}} />
  </colgroup>

  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Model</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Quantization</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Weights</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Qwen3.8-27B</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>BF16</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen/Qwen3.8-27B</a></td>
    </tr>

    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.05)"}}>Qwen3.8-27B-FP8</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>FP8 (blockwise)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><a href="https://huggingface.co/Qwen/Qwen3.8-27B-FP8">Qwen/Qwen3.8-27B-FP8</a></td>
    </tr>

    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Qwen3.8-27B-NVFP4 (FP4 head)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>NVFP4 W4A4 + FP8 projections, `lm_head` packed to FP4</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><a href="https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4">RadixArk/Qwen3.8-27B-NVFP4</a></td>
    </tr>

    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.05)"}}>Qwen3.8-27B-NVFP4 (BF16 head)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Same body, `lm_head` left dense in BF16</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><a href="https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead">RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead</a></td>
    </tr>

    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Qwen3.8-27B-NVFP4 (NVIDIA)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>NVIDIA's ModelOpt export of the same W4A4 body, `lm_head` packed to FP4</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><a href="https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4">nvidia/Qwen3.8-27B-NVFP4</a></td>
    </tr>
  </tbody>
</table>

The two RadixArk NVFP4 exports differ only in the `lm_head`: one packs it to
FP4, the other leaves it dense in BF16. The dense head is \~1.7 GB larger on disk
and \~3.2 GB larger at runtime, so it is the harder of the two to fit — every
recipe on this page was measured against it, and the FP4-head cells reuse
those pins unchanged.

NVIDIA's own export is that same W4A4 body with that same FP4 head: identical
quantized-layer map (FP8 attention and GDN projections, NVFP4 MLPs), identical
tensor set, identical 21.9 GB on disk. On GB300, RTX PRO 6000 and DGX Spark its
cells reuse the FP4-head pins unchanged, and both SM12x grids have been
re-measured against this export on v0.5.19: all 16 overlay combinations per
card serve and score 94.01-95.00% (RTX PRO 6000) and 94.16-95.07% (DGX Spark)
on the full 1319-question GSM8K.

The RTX 5090 is measured too — all 15 overlay combinations it offers serve and
score 93.93-94.92% — and every winning launch command there is identical to the
FP4-head export's, which is the strongest form of the claim above. What the
32GB card does need is the draft-model rows pinning their own pools: those
recipes pin `--max-running-requests 1`, but nothing caps the pools to match, so
the KV pool sizes itself for 127,332 tokens against the 9,216 one
8192-in/1024-out request needs, and the engine's default split then leaves the
GDN state pool far short of the slots it needs once the draft model's weights
are counted against `--mem-fraction-static`. The DSPARK row therefore pins
`--max-total-tokens` and a measured `--mamba-full-memory-ratio`, as do DFLASH2
and MTP on the dense-lm\_head export at float32 state. Those pins override the
calculator's live value for the selections that carry them. The no-speculation
row needs none of it and runs at the pins shown.

The two RadixArk checkpoints declare `kv_cache_quant_algo: FP8`, so SGLang's
default `--kv-cache-dtype auto` already puts their KV pool in `fp8_e4m3`. The
NVIDIA export ships no `kv_cache_scheme`, so `auto` would leave its pool in
BF16 instead. Every recipe on this page pins `--kv-cache-dtype fp8_e4m3`
explicitly, so all three run the same `fp8_e4m3` pool regardless; the
difference only shows up if you switch the Playground's **KV Cache Precision**
row back to Auto.

## 2. Configuration Tips

* **SM120/SM121 (RTX PRO 6000 Blackwell, RTX 5090, DGX Spark)**: use `--attention-backend
  flashinfer`; `trtllm_mha` is SM100-only. MTP with the FlashInfer backend
  requires a FlashInfer build whose prefill `plan` accepts `uniform_q_len`
  (newer than 0.6.15.post1); otherwise run spec with `--attention-backend triton`.
  On DGX Spark the 128GB is unified memory shared with the host CPU, so all
  three checkpoints fit, and its cells reuse the RTX PRO 6000 recipe at
  `--mem-fraction-static 0.80` rather than a separate operating point. The one
  lower pin is the unified pool pricing the host's memory too: 0.85 of 128GB
  leaves \~8GB for the OS — exactly DGX OS earlyoom's SIGTERM threshold — and
  the first long prefill or boot-time graph capture dips under it and gets the
  scheduler killed with `exit code -15` and no traceback (`journalctl -u
  earlyoom` shows the kill). At 0.85, 15 of the 48 cells were killed that way,
  and which cells is margin noise; at 0.80 every cell served on every attempt.
  **Validated on SM121 / aarch64**: all 80 configurations (5 checkpoints x
  Speculative Decoding x Serving Strategy x Mamba SSM Dtype, DFLASH2 included)
  served on GB10 on `v0.5.19` at ISL 8192 / OSL 1024, concurrency 1, and each
  scored the full 1319-question GSM8K (93.18-95.15%); the float32 and bfloat16
  halves ran on two separate GB10 boxes. No throughput or acceptance-length
  numbers were re-taken. The sweep exercises the FlashInfer `plan` /
  `uniform_q_len` path above, which raised no arity error on that build. Three host quirks when reproducing
  on GB10: docker GPU access is CDI-only (`--device nvidia.com/gpu=all`, as no
  `nvidia` runtime is registered); `nvidia-smi` reports `Not Supported` for
  memory because it is unified with the CPU — gate a relaunch on `MemAvailable`
  in `/proc/meminfo` instead; and the BF16 checkpoint takes \~6.5 minutes just
  to load its 18 shards from NVMe, so budget \~10 minutes to READY before
  calling a boot hung.
* **H200 (SM90)**: BF16 and FP8 only — the card has no FP4 tensor cores, so an
  NVFP4 checkpoint's MLP would fall back to the Marlin W4A16 weight-only path,
  and all three NVFP4 cells are greyed out. The H200 recipes use 32768-token
  prefill chunks (SM90 prefill is fast enough that a big chunk barely stalls
  decode, unlike the SM120 guidance below), and the FlashInfer GDN prefill
  backend engages by default under them. `--attention-backend fa3` is a valid
  alternative, measured slightly faster at bs=1.
* **MTP**: `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` uses the
  in-checkpoint MTP head. (This recipe was originally documented with `NEXTN`,
  an alias of `EAGLE` — same algorithm.)
* **DSpark**: the trained draft model is a separate checkpoint — add
  `--speculative-algorithm DSPARK --speculative-draft-model-path
  RadixArk/Qwen3.8-27B-DSpark` (the Playground's Speculative Decoding card
  emits this pair). DSpark does **not** take
  `--speculative-num-draft-tokens`: its verify window is
  `--speculative-dspark-block-size` (gamma) **+ 1**, and gamma is auto-inferred
  from the draft checkpoint when the flag is omitted (7 for this checkpoint, so
  D = 8). That `D` is a term in the balanced ratio —
  `r = (S + D) x token_equiv / L`, where `token_equiv` is the state slot
  expressed in KV tokens, `state_bytes / kv_bytes_per_token` (4698 at fp32
  state / 2394 at bf16, over fp8 KV) — so DSpark needs a materially higher
  `--mamba-full-memory-ratio` than no-spec at the same `S`, and pinning a
  different gamma changes the ratio with it. MTP is the opposite case: with
  `--enable-linear-replayssm-spec` its draft intermediates move onto a fixed
  ring, so `D = 0` and the ratio returns to the no-spec value. The
  [calculator](#mamba-ratio-calculator) applies both rules.
* **DFlash2**: a trained block-diffusion draft in a separate checkpoint — add
  `--speculative-algorithm DFLASH --speculative-draft-model-path
  incoai/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8` (8 is the
  draft's block size, and it is the `D` term in the ratio, same value as
  DSpark's). Runs on Ascend NPUs as well
  ([#35629](https://github.com/sgl-project/sglang/pull/35629)): the selector
  verify falls back to argmax there, matching what EAGLE and first-generation
  DFlash drafts (such as `z-lab/Qwen3-8B-DFlash-b16`) already do on NPU.
  Therefore, NPU currently guarantees lossless verification only for greedy
  requests; use `temperature=0` and `top_k=1`. Non-greedy requests log a warning;
  both draft proposal and target verification fall back to greedy, so their
  requested sampling distribution is not preserved. The selector projects
  candidates through the target `lm_head`,
  including quantized heads, so it runs on the NVFP4 checkpoint (whose head
  is NVFP4-packed; the BF16 and FP8 checkpoints keep a dense head).
  The Ascend comparison in #35629 used an A3 Series device with BF16 target weights,
  `--tp-size 2 --attention-backend ascend --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer`, and disabled RadixCache for both
  baseline and DFlash2 to exclude cache warm-up and prefix reuse. The DFlash2
  run added the three flags shown above.
  That comparison's accuracy used zero-shot GSM8K with greedy sampling,
  `max_new_tokens=2048`, 128 examples, and concurrency levels 1, 2, 4, 8
  and 16 — a different protocol from this page's own sweep below.
  Validation: every SM12x cell on this page is measured end to end on
  v0.5.19 — 202 cells over the five checkpoints, four speculative options, two
  serving tiers and two GDN state dtypes, full 1319-question GSM8K on each,
  93.18-95.15%. The RTX PRO 6000 and DGX Spark recipes need no changes. On the
  32GB RTX 5090 the panel applies the measured pins automatically: DFlash2 at
  `--mem-fraction-static 0.91` with `--chunked-prefill-size 1024` — at 0.91 the
  pools fit but a 2048-token chunk's activations do not — DSpark at 0.88
  (bfloat16), 0.91 (float32) and 0.92 on the dense-lm\_head export, all three
  with their pools pinned and the last two also cutting the prefill chunk to
  1024 and 512, EAGLE at 0.93 (bfloat16) and 0.94 (float32), and
  no-speculation at 0.90.
  Whether float32 is available with a draft model depends on the `lm_head`: on
  the BF16-head export it is greyed out for both DSpark and DFlash2, since the
  dense head's \~3.2 GB leave no fp32 state pool that also clears prefill graph
  capture. The FP4-head export frees that headroom back — DSpark serves at 0.91
  and DFlash2 High-Throughput at 0.895 with `--mamba-full-memory-ratio 10`
  overriding the balanced value — and only DFlash2 Low-Latency stays out of
  reach, where five fp32 slots and a full request's KV never coexist. bfloat16
  remains the faster choice regardless: DFlash2 posts 4.92 ms median TPOT at an
  accept length of 4.29, the best result on this card.
* **Hardware fit**: FP8 weights \~28.5GB (not serviceable beyond bs≤2 on
  32GB cards); NVFP4 weights \~16.5GB (recommended for RTX 5090-class GPUs).
* `--mamba-radix-cache-strategy extra_buffer_lazy` lowers the state cost per
  request from 5 slots to 4 at no accuracy cost. On small-VRAM cards (RTX 5090
  32GB) the state pool bounds concurrency long before KV does — prefer lowering
  `S` (lazy strategy, or `--disable-radix-cache` for S=1); the
  [calculator](#mamba-ratio-calculator) re-derives the ratio for the new `S`.
  The balanced ratio itself is VRAM-independent.
* `--mamba-ssm-dtype`: the GDN state slot is **153.9 MB at `float32`** (the
  checkpoint's declared precision) and **78.4 MB at `bfloat16`**, so bf16 roughly
  halves the state pool and hands the difference to KV — measured on an RTX 5090
  with no speculation, 97,280 KV tokens at bf16 against 68,588 at fp32. On 32GB
  cards it also decides whether a config fits at all: EAGLE needs
  `--mem-fraction-static 0.94` at fp32 but 0.92 at bf16. Speed is **not** a
  one-way trade — with speculative decoding fp32 sometimes wins (NVFP4 + EAGLE:
  152.9 vs 144.5 tok/s/user) and sometimes loses (FP8 + EAGLE: 106.3 vs 116.1);
  measure both for your quantization. Treat
  `bfloat16` as an accuracy gate and validate it for your workload. On SM120
  both precisions run the Triton linear-attn prefill path — the FlashInfer GDN
  prefill fast path gates on SM100, where its validated domain is in fact a
  bf16 state pool — so no dtype forces an extra flag here. One interaction to
  know: `--enable-linear-replayssm-spec` auto-selects fp32 state when
  `--mamba-ssm-dtype` is unset, and an explicit non-fp32 value logs a
  state-drift warning at boot. The SSM dtype row always emits the flag
  explicitly, so the bf16 + EAGLE cells run with that warning — accounted for
  in their validation.
* `--chunked-prefill-size 2048`: decode steps stall behind each prefill chunk
  on hybrid GDN models, and 8192-token chunks stall them \~600ms at a time.
  2048 keeps decode inter-token latency smooth under mixed load and also
  improves single-wave TTFT.

## 3. Agent Harnesses

Agent harnesses drive the model through the OpenAI-compatible endpoint — or, for
Claude Code, through SGLang's Anthropic-compatible one — so any of them works
once three things line up.

**The parsers ship in the command.** Every recipe above carries
`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`, because without them a
harness receives tool calls as raw text instead of structured `tool_calls`. The
**Parsers** card in the [Playground](#playground) is therefore an opt-out — both
chips start on, and turning one off strips its flag.

`qwen3_coder` is the right tool-call parser for this checkpoint: its chat
template instructs the model to reply with an inner `<function=…>` /
`<parameter=…>` block nested in `<tool_call></tool_call>`, which is exactly what
that parser decodes. The Hermes parser (`--tool-call-parser hermes`) reads a
*different* payload — bare JSON inside `<tool_call>` — so pointing a Hermes-format
harness at this model without switching the flag yields tool calls that never
parse. `--reasoning-parser qwen3` matches the template's `enable_thinking`
toggle, which defaults to on.

**Endpoint and model id.** The base URL is `http://<host>:30000/v1`. The `model`
string a harness sends must equal the server's `--model-path` — the OpenAI
`/v1/models` name defaults to it — unless you override it with
`--served-model-name`, which is usually worth doing to keep harness configs short.

SGLang also serves an Anthropic-compatible `/v1/messages`, which is what
[§3.3](#3-3-claude-code) uses. It converts each request to the OpenAI shape,
hands it to the same chat-serving path, and converts the response back — so the
parser flags above apply there identically.

**Auth.** `--api-key` is unset by default, so the server accepts unauthenticated
requests. Harnesses that insist on a key can send any placeholder; set
`--api-key` on the server if the endpoint is reachable beyond localhost.

### 3.1 OpenCode

[OpenCode](https://opencode.ai/docs/providers/) reaches a self-hosted endpoint
through a provider entry in `opencode.json`.

<Accordion title="Register SGLang as an OpenCode provider">
  Store the credential first — pick **Other**, give the provider an id, and enter
  any placeholder when the server has no `--api-key`:

  ```bash Command theme={null}
  opencode
  /connect
  ```

  Then declare the provider in `opencode.json`:

  ```json Config theme={null}
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "sglang": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "SGLang (Qwen3.8-27B)",
        "options": {
          "baseURL": "http://localhost:30000/v1"
        },
        "models": {
          "RadixArk/Qwen3.8-27B-NVFP4": {
            "name": "Qwen3.8-27B NVFP4"
          }
        }
      }
    }
  }
  ```

  `npm` selects the transport — `@ai-sdk/openai-compatible` is the one for a plain
  OpenAI-shaped endpoint. `apiKey` is optional and takes a `"{env:VAR_NAME}"`
  reference rather than a literal. The `models` keys are the ids sent on the wire,
  so they must match the served model name. Confirm with `/models`.
</Accordion>

### 3.2 Pi

[Pi](https://pi.dev/docs/latest/custom-provider)
(`@earendil-works/pi-coding-agent`) registers providers from an extension rather
than a config file.

<Accordion title="Register SGLang as a Pi provider">
  ```javascript Extension theme={null}
  pi.registerProvider("sglang", {
    baseUrl: "http://localhost:30000/v1",
    api: "openai-completions",
    apiKey: "$SGLANG_API_KEY",
    models: [
      {
        id: "RadixArk/Qwen3.8-27B-NVFP4",
        name: "Qwen3.8-27B",
        reasoning: true,
        input: ["text", "image"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 262144,
        maxTokens: 32768,
      },
    ],
  });
  ```

  `api: "openai-completions"` is what selects the OpenAI-compatible transport, and
  `apiKey` takes a `$ENV_VAR` reference rather than a literal. `contextWindow` is
  the checkpoint's native 262,144; set `maxTokens` to whatever output cap you want
  per turn. Confirm registration with `pi --list-models`.
</Accordion>

### 3.3 Claude Code

Claude Code speaks the Anthropic API, so it points at SGLang's `/v1/messages`
rather than the OpenAI endpoint.

<Warning>
  Anthropic documents that routing Claude Code to non-Claude models through a
  gateway is **not supported**. The wiring below works because SGLang implements
  the Anthropic message format, but it sits outside what Claude Code is tested
  against — expect newer Claude Code features to degrade or fail.
</Warning>

<Accordion title="Point Claude Code at SGLang">
  `ANTHROPIC_BASE_URL` is the server origin — Claude Code appends `/v1/messages`
  itself, so leave the `/v1` suffix off:

  ```bash Command theme={null}
  export ANTHROPIC_BASE_URL=http://localhost:30000
  export ANTHROPIC_AUTH_TOKEN=placeholder
  ```

  The two credential variables travel in different headers:
  `ANTHROPIC_AUTH_TOKEN` goes out as `Authorization: Bearer`, `ANTHROPIC_API_KEY`
  as `x-api-key`. Either satisfies a server started without `--api-key`; with
  `--api-key` set, pick the variable matching the header your server reads. A
  credential variable also takes precedence over a saved claude.ai login for that
  session.

  The same pair can live in a settings file instead, which persists across shells
  and wins over a shell export:

  ```json Config theme={null}
  {
    "env": {
      "ANTHROPIC_BASE_URL": "http://localhost:30000",
      "ANTHROPIC_AUTH_TOKEN": "placeholder"
    }
  }
  ```

  Run `/status` in Claude Code to confirm which base URL and credential source the
  session picked up.
</Accordion>

### 3.4 Hermes Agent

[Hermes Agent](https://github.com/NousResearch/hermes-agent) (Nous Research, MIT)
selects a self-hosted endpoint through its setup wizard or its config file.

<Accordion title="Point Hermes Agent at SGLang">
  ```bash Command theme={null}
  hermes model
  # choose "Custom endpoint (self-hosted / VLLM / etc.)", then enter the
  # base URL, an API key (blank for a local server) and the model name
  ```

  Equivalently, in `~/.hermes/config.yaml`:

  ```yaml Config theme={null}
  model:
    default: RadixArk/Qwen3.8-27B-NVFP4
    provider: custom
    base_url: http://localhost:30000/v1
    api_key: ""
    context_length: 262144
  ```

  For several endpoints at once, declare them under `providers:` and switch with
  `/model custom:<name>` mid-session:

  ```yaml Config theme={null}
  providers:
    workstation:
      api: http://localhost:30000/v1
    server:
      api: https://gpu-host.internal:30000/v1
      key_env: SGLANG_API_KEY
  ```
</Accordion>

## 4. Semantic Decisions

Qwen3.8-27B can answer typed questions about an input with a probability for every option through `/v1/decisions`, without generating text. The server turns thinking off for these requests, so the answer is read after a closed think block, and the same server keeps serving chat traffic. No launch flag is needed. Until a release contains `/v1/decisions`, install a [nightly build](/docs/get-started/install#nightly-builds). With a server started by `sglang serve --model-path Qwen/Qwen3.8-27B --host 127.0.0.1 --port 30000`:

```python Example theme={null}
import requests

response = requests.post(
    "http://127.0.0.1:30000/v1/decisions",
    json={
        "input": "I've been trying to connect my Stripe account for 3 days and the integration keeps failing.",
        "questions": [
            {
                "id": "team",
                "type": "choice",
                "question": "Which team should handle this ticket?",
                "options": [{"name": "billing"}, {"name": "technical"}, {"name": "sales"}],
            },
            {"id": "urgent", "type": "yes_no", "question": "The customer needs an answer today."},
        ],
    },
    timeout=60,
)
response.raise_for_status()
answers = response.json()["answers"]
print(answers["team"]["choice"], answers["team"]["probabilities"])
print(answers["urgent"]["probabilities"]["yes"])
```

See [Decision models](/docs/supported-models/decision_models) for the request reference, how the probabilities are computed, and the limits.

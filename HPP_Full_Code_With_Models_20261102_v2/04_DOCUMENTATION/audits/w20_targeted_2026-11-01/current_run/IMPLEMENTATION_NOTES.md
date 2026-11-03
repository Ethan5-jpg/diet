# Current run implementation and provenance notes

Only Chrome's claimed 5bc05bbe… AppStream tab was operated. New entry point: installed Chrome skill at `/Applications/ChatGPT.app/Contents/Resources/plugins/openai-bundled/plugins/chrome/skills/control-chrome/SKILL.md`, using the current node-repl browser runtime; old cua_repl was not used. The current Chrome connection succeeded even though the previous chat had a browser-control error. User screenshot showed 14054… but the initial explicit project boundary and verified 5bc05bbe… terminal were followed.

Every UI mutation was followed by an explicit >=1500 ms pause and state/screenshot observation. Clipboard text was our known generated code/command; AppStream Clipboard Meta+v transferred to the remote clipboard, then MATE Edit/Paste inserted it, then separate Return executed it. Panels can auto-close; no session refresh/restart. AX screenshot does not refresh AX element grounding: get fresh AX state before using a toolbar index again. A stale index produced `Node is detached from document`; fresh state fixed it without reconnecting.

47 existing preflight was read and actually run. v2 source dependencies and protected/critical artifact hashes were checked; v3 directories did not exist. pheno Python path was actually printed. Remote `rg` was unavailable, so subsequent remote status checks used standard grep/Python; local file search used rg.

48 registered the full estimand and finite selection recipe before new model fits. Its four checks passed both locally and in TRE. Source hashes are in per-transfer manifests, expected_v3_sources JSONs and actual remote logs. All remote files were written through the browser UI with exclusive creation and verified hashes.

49 first attempt failed at saving the first fold: mounted filesystem does not support the seek needed by direct `np.savez_compressed(path,...)`. Preserved source/log/PID/exit1 and partial npz. 49b changes only output path and serialization: BytesIO -> sequential write, results/attempt49b. It exited0 and preserved baseline predictions within 2e-9. 50 was transferred only; 50b used the same write correction and actually completed daily analysis, exit0.

51 initial independent verifier rejected object-dtype household strings when loading daily regression npz with allow_pickle=False. Preserved exit1/log. 51b reads only numeric arrays and obtains grouping labels from verified daily Parquet, then independently recalculates WLS and household sandwich. No allow_pickle=True, altered regression or changed statistical rule.

Model result and daily scientific claims must be taken from the actual successful 49b/50b outputs and final verified readbacks, not inferred from local code. Local deliverables contain generic source, aggregate manually transcribed values and screenshots only; no individual parquet/npz/data were downloaded.

52 report append was also unsupported by the mount (exit1); 52b wrote a fresh FINAL_REPORT.md and sealed 43 artifacts (exit0). Scientific model and daily results were not rerun or changed. All final formula/date/weight/covariance checks passed.

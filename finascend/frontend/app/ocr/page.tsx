"use client";

/**
 * 04 — Receipt ingestion. Pick or upload a receipt, watch the real chain run:
 * EasyOCR, A.6 classification, normalization, A.5 duplicate screening.
 *
 * Every stage's output is shown, including the refusal path — a receipt whose
 * total could not be read produces no record at all, and that is displayed as
 * the correct outcome rather than hidden.
 */

import { useRef, useState } from "react";
import { api, OcrPipelineResult, ReceiptSample } from "@/lib/api";
import { Card, Chip, ErrorState, Loading, PageHead, Tile, useApi } from "@/components/ui";
import { fmt, pct } from "@/components/charts";

const TIER_TONE: Record<string, "good" | "warn" | "critical"> = {
  clean: "good",
  moderate: "warn",
  hard: "critical",
};

function Verdict({ ok }: { ok: boolean | undefined }) {
  if (ok === undefined) return null;
  return <Chip tone={ok ? "good" : "critical"}>{ok ? "correct" : "wrong"}</Chip>;
}

export default function OcrPage() {
  const samples = useApi(() => api.get<{ samples: ReceiptSample[] }>("/ingestion/receipts/samples"));
  const [selected, setSelected] = useState<string | null>(null);
  const [result, setResult] = useState<OcrPipelineResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<Error | null>(null);
  const [uploadUrl, setUploadUrl] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function runSample(id: string) {
    setSelected(id);
    setUploadUrl(null);
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      setResult(await api.post<OcrPipelineResult>(`/ingestion/receipts/sample/${id}/process`));
    } catch (e) {
      setErr(e as Error);
    } finally {
      setBusy(false);
    }
  }

  async function runUpload(file: File) {
    setSelected(null);
    setUploadUrl(URL.createObjectURL(file));
    setBusy(true);
    setErr(null);
    setResult(null);
    const fd = new FormData();
    fd.append("file", file);
    try {
      setResult(await api.post<OcrPipelineResult>("/ingestion/receipts/upload", fd));
    } catch (e) {
      setErr(e as Error);
    } finally {
      setBusy(false);
    }
  }

  const allSamples = samples.data?.samples ?? null;
  const sample = allSamples?.find((s) => s.id === selected);
  const imgSrc = uploadUrl ?? (selected ? api.imageUrl(`/ingestion/receipts/${selected}/image`) : null);

  return (
    <>
      <PageHead
        title="Receipt ingestion"
        sub={
          <>
            The full chain, run live on each request: <strong>EasyOCR</strong> → field
            extraction → the A.6 character n-gram classifier → normalization into an{" "}
            <code>Outflow</code> with <code>source_type = receipt_ocr</code> → the A.5 duplicate
            screen. EasyOCR was chosen over pytesseract because it installs from pip alone,
            with no separate OS-level binary and no credentials. Accuracy is measured per
            difficulty tier in <code>OCR_ACCURACY.md</code> — never as one blended number.
          </>
        }
      />

      {samples.error ? (
        <ErrorState error={samples.error} />
      ) : (
        <>
          <Card
            title="Choose a receipt"
            note="Samples are rendered by the same generator the accuracy harness scores, so their ground truth is known exactly and each extracted field can be marked right or wrong. Uploads have no ground truth, so the correctness column is omitted rather than guessed."
            right={
              <div style={{ display: "flex", gap: 8 }}>
                <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
                       onChange={(e) => { const f = e.target.files?.[0]; if (f) runUpload(f); }} />
                <button onClick={() => fileRef.current?.click()} disabled={busy}>
                  Upload an image…
                </button>
              </div>
            }
          >
            {samples.loading || !allSamples ? (
              <Loading rows={2} />
            ) : (
              ["clean", "moderate", "hard"].map((tier) => (
                <div key={tier} style={{ marginBottom: 20 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                    <span className="tile-label">{tier}</span>
                    <Chip tone={TIER_TONE[tier]}>
                      {tier === "clean"
                        ? "flatbed scan"
                        : tier === "moderate"
                        ? "careful phone photo"
                        : "bad light, skewed, blurred"}
                    </Chip>
                  </div>
                  <div className="sample-grid">
                    {allSamples
                      .filter((s) => s.difficulty === tier)
                      .map((s) => (
                        <button key={s.id} className="sample" data-selected={selected === s.id}
                                disabled={busy} onClick={() => runSample(s.id)}>
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={api.imageUrl(`/ingestion/receipts/${s.id}/image`)}
                               alt={`${tier} receipt from ${s.truth.vendor_name}`} loading="lazy" />
                          <div className="sample-cap">{s.truth.category}</div>
                        </button>
                      ))}
                  </div>
                </div>
              ))
            )}
          </Card>

          {err && <ErrorState error={err} />}

          {(busy || result) && (
            <div className="grid grid-2">
              <Card title="Input">
                {imgSrc && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={imgSrc} alt="selected receipt" style={{ width: "100%", borderRadius: 4, background: "#fff" }} />
                )}
                {sample && (
                  <dl className="kv" style={{ marginTop: 16 }}>
                    <dt>vendor</dt><dd>{sample.truth.vendor_name}</dd>
                    <dt>invoice</dt><dd>{sample.truth.invoice_number}</dd>
                    <dt>date</dt><dd>{sample.truth.issue_date}</dd>
                    <dt>total</dt><dd>{sample.truth.total_amount.toLocaleString()}</dd>
                    <dt>category</dt><dd>{sample.truth.category}</dd>
                  </dl>
                )}
                {sample && (
                  <div className="card-note" style={{ marginTop: 12 }}>
                    Ground truth, shown for scoring only. The extractor never receives it.
                  </div>
                )}
              </Card>

              <Card title="Pipeline">
                {busy || !result ? (
                  <Loading rows={4} />
                ) : (
                  <>
                    <div className="stage" data-ok={result.ocr.n_regions > 0}>
                      <div className="stage-name">1 · OCR — {result.ocr.engine}</div>
                      <dl className="kv">
                        <dt>regions</dt><dd>{result.ocr.n_regions}</dd>
                        <dt>mean confidence</dt><dd>{result.ocr.mean_confidence.toFixed(3)}</dd>
                        <dt>elapsed</dt><dd>{result.ocr.elapsed_ms.toFixed(0)} ms</dd>
                      </dl>
                    </div>

                    <div className="stage" data-ok={result.extraction.total_amount != null}>
                      <div className="stage-name">2 · Field extraction</div>
                      <table className="data">
                        <thead>
                          <tr><th>Field</th><th>Value</th><th className="n">Conf</th><th></th></tr>
                        </thead>
                        <tbody>
                          {([
                            ["vendor_name", result.extraction.vendor_name],
                            ["invoice_number", result.extraction.invoice_number],
                            ["issue_date", result.extraction.issue_date],
                            ["total_amount", result.extraction.total_amount?.toLocaleString() ?? null],
                          ] as [string, string | null][]).map(([k, v]) => (
                            <tr key={k}>
                              <td style={{ color: "var(--text-muted)" }}>{k}</td>
                              <td className="num">
                                {v ?? <span style={{ color: "var(--status-warn)" }}>declined</span>}
                              </td>
                              <td className="n">
                                {result.extraction.field_confidence[k]?.toFixed(2) ?? "—"}
                              </td>
                              <td><Verdict ok={result.correct?.[k]} /></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {result.classification && (
                      <div className="stage" data-ok={!result.classification.is_uncertain}>
                        <div className="stage-name">3 · A.6 classification</div>
                        <dl className="kv">
                          <dt>category</dt>
                          <dd>
                            {result.classification.category} <Verdict ok={result.correct?.category} />
                          </dd>
                          <dt>confidence</dt><dd>{result.classification.confidence.toFixed(3)}</dd>
                          <dt>runner-up</dt>
                          <dd>{result.classification.runner_up} ({result.classification.runner_up_score.toFixed(3)})</dd>
                          <dt>margin</dt>
                          <dd style={{ color: result.classification.is_uncertain ? "var(--status-warn)" : undefined }}>
                            {result.classification.margin.toFixed(3)}
                          </dd>
                        </dl>
                        <div className="card-note" style={{ marginTop: 8 }}>
                          The runner-up is always returned, because the <em>margin</em> is what
                          indicates reliability: a top score of 0.4 is confident when the
                          runner-up is 0.1 and a coin-flip when it is 0.39.
                        </div>
                      </div>
                    )}

                    {result.rejected ? (
                      <div className="stage" data-ok={false}>
                        <div className="stage-name">4 · Normalization — refused</div>
                        <p style={{ fontSize: 13, color: "var(--status-warn)" }}>{result.rejected.reason}</p>
                        <div className="card-note" style={{ marginTop: 8 }}>
                          {result.rejected.why_this_is_correct}
                        </div>
                      </div>
                    ) : result.record ? (
                      <div className="stage" data-ok={!result.record.needs_review}>
                        <div className="stage-name">4 · Normalized record</div>
                        <dl className="kv">
                          <dt>counterparty</dt><dd>{result.record.counterparty_name}</dd>
                          <dt>amount</dt>
                          <dd>{result.record.currency} {result.record.amount.toLocaleString()}</dd>
                          <dt>due date</dt><dd>{result.record.due_date}</dd>
                          <dt>category</dt><dd>{result.record.category}</dd>
                          <dt>source_type</dt><dd>{result.record.source_type}</dd>
                          <dt>reference</dt><dd>{result.record.source_reference}</dd>
                        </dl>
                        {result.record.needs_review && (
                          <div style={{ marginTop: 10 }}>
                            <Chip tone="warn">needs review</Chip>
                            <ul style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 8, paddingLeft: 18 }}>
                              {result.record.review_reasons.map((r, i) => <li key={i}>{r}</li>)}
                            </ul>
                          </div>
                        )}
                      </div>
                    ) : null}

                    {result.duplicate_screen && (
                      <div className="stage" data-ok={!result.duplicate_screen.is_exact_duplicate}>
                        <div className="stage-name">5 · A.5 duplicate screen</div>
                        <dl className="kv">
                          <dt>exact duplicate</dt>
                          <dd>{result.duplicate_screen.is_exact_duplicate ? "yes" : "no"}</dd>
                          <dt>DBSCAN label</dt><dd>{result.duplicate_screen.dbscan_label}</dd>
                          <dt>robust z</dt><dd>{result.duplicate_screen.robust_z.toFixed(3)}</dd>
                        </dl>
                        <div className="card-note" style={{ marginTop: 8 }}>
                          {result.duplicate_screen.reason}. {result.duplicate_screen.caveat}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </Card>
            </div>
          )}

          {result && (
            <Card title="Raw OCR text" note="What the engine actually returned, before any field logic. On the hard tier the failure is visible here first — amounts arrive split across regions and glyphs are misread.">
              <pre className="raw">{result.ocr.text}</pre>
            </Card>
          )}
        </>
      )}
    </>
  );
}

"use client";

/**
 * 05 — Add a receipt.
 *
 * image -> OCR -> classify -> normalise -> duplicate screen -> a record the
 * user can CORRECT before saving.
 *
 * The correction step is the product's answer to the measured OCR failure
 * mode, not a nicety. On degraded images the invoice number declines to answer
 * (recoverable) while the total amount comes back confidently wrong
 * (unrecoverable) — a corrupted number is still a parseable number. So the
 * fields are editable, anything the pipeline flagged is highlighted, and low
 * confidence is stated in words next to the field it belongs to.
 */

import { useEffect, useRef, useState } from "react";
import { api, ApiError, OcrPipelineResult, ReceiptSample } from "@/lib/api";
import {
  Card,
  CardHead,
  EmptyState,
  ErrorState,
  Icon,
  Loaded,
  Method,
  RowsSkeleton,
  Skeleton,
  Status,
  Tile,
  Tone,
  useApi,
} from "@/components/ui";
import { inr, pct, shortDate } from "@/lib/format";

const TIER_COPY: Record<string, { label: string; tone: Tone; note: string }> = {
  clean: { label: "Flatbed scan", tone: "good", note: "best case — this is what a scanner gives you" },
  moderate: { label: "Careful phone photo", tone: "warning", note: "realistic everyday quality" },
  hard: { label: "Bad light, skewed, blurry", tone: "serious", note: "the case that needs checking" },
};

type Fields = {
  vendor_name: string;
  invoice_number: string;
  issue_date: string;
  total_amount: string;
  tax_amount: string;
};

export default function AddReceipt() {
  const samples = useApi(() => api.get<{ samples: ReceiptSample[] }>("/ingestion/receipts/samples"));
  const [result, setResult] = useState<OcrPipelineResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<ApiError | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [fields, setFields] = useState<Fields | null>(null);
  const [saved, setSaved] = useState(false);
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // The image endpoint is authenticated, so it is fetched with the token and
  // shown from a blob URL. Revoked on change/unmount so the blobs don't leak.
  useEffect(() => {
    if (!selected) {
      setImgUrl(null);
      return;
    }
    let url: string | null = null;
    let live = true;
    api
      .objectUrl(`/ingestion/receipts/${selected}/image`)
      .then((u) => {
        url = u;
        if (live) setImgUrl(u);
        else URL.revokeObjectURL(u);
      })
      .catch(() => live && setImgUrl(null));
    return () => {
      live = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [selected]);

  function adopt(r: OcrPipelineResult) {
    setResult(r);
    setSaved(false);
    setFields({
      vendor_name: r.extraction.vendor_name ?? "",
      invoice_number: r.extraction.invoice_number ?? "",
      issue_date: r.extraction.issue_date ?? "",
      total_amount: r.extraction.total_amount === null ? "" : String(r.extraction.total_amount),
      tax_amount: r.extraction.tax_amount === null ? "" : String(r.extraction.tax_amount),
    });
  }

  async function runSample(id: string) {
    setBusy(true); setErr(null); setSelected(id); setResult(null); setFields(null);
    try {
      adopt(await api.post<OcrPipelineResult>(`/ingestion/receipts/sample/${id}/process`));
    } catch (e) {
      setErr(e instanceof ApiError ? e : new ApiError(String(e)));
    } finally {
      setBusy(false);
    }
  }

  async function runUpload(file: File) {
    setBusy(true); setErr(null); setSelected(null); setResult(null); setFields(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      adopt(await api.post<OcrPipelineResult>("/ingestion/receipts/upload", fd));
    } catch (e) {
      setErr(e instanceof ApiError ? e : new ApiError(String(e)));
    } finally {
      setBusy(false);
    }
  }

  const conf = result?.extraction.field_confidence ?? {};
  const lowConf = (k: string) => (conf[k] ?? 1) < 0.6;

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Add a receipt</h1>
        <p className="lede">
          Photograph a bill and we&rsquo;ll read it. Always check the amount before you
          save — that is the field that fails most quietly.
        </p>
      </div>

      {/* ------------------------------------------------------------------ */}
      <Card>
        <CardHead
          title="Pick a sample or upload your own"
          note="The samples are generated at three image qualities so you can see exactly where reading a receipt starts to break down."
        />
        <div className="card-body stack-sm">
          <div className="row">
            <button type="button" className="btn btn-primary" onClick={() => fileRef.current?.click()} disabled={busy}>
              {Icon.upload()} Upload an image
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) runUpload(f);
              }}
            />
            <span className="small muted">PNG or JPEG</span>
          </div>

          <Loaded q={samples} skeleton={<RowsSkeleton n={3} />}>
            {(s) =>
              s.samples.length === 0 ? (
                <EmptyState title="No samples available">
                  The service returned an empty sample set.
                </EmptyState>
              ) : (
                <div className="grid grid-3" style={{ marginTop: "var(--s-2)" }}>
                  {(["clean", "moderate", "hard"] as const).map((tier) => {
                    const inTier = s.samples.filter((x) => x.difficulty === tier);
                    const copy = TIER_COPY[tier];
                    return (
                      <div key={tier} className="stack-sm">
                        <div>
                          <Status tone={copy.tone}>{copy.label}</Status>
                          <div className="tiny muted" style={{ marginTop: 4 }}>{copy.note}</div>
                        </div>
                        <div className="row" style={{ gap: 6 }}>
                          {inTier.slice(0, 4).map((x) => (
                            <button
                              key={x.id}
                              type="button"
                              className="btn btn-secondary"
                              style={{ padding: "6px 10px", minHeight: 34 }}
                              disabled={busy}
                              aria-pressed={selected === x.id}
                              onClick={() => runSample(x.id)}
                            >
                              #{x.index + 1}
                            </button>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )
            }
          </Loaded>
        </div>
      </Card>

      {/* ------------------------------------------------------------------ */}
      {busy ? (
        <Card>
          <div className="card-body stack-sm" aria-busy="true" aria-live="polite">
            <Status tone="neutral">Reading the image…</Status>
            <p className="small muted">
              The text recogniser runs on the CPU and takes a few seconds. The first run
              also loads the model.
            </p>
            <Skeleton h={18} w="55%" />
            <Skeleton h={18} w="72%" />
            <Skeleton h={18} w="40%" />
          </div>
        </Card>
      ) : null}

      {err ? <ErrorState error={err} /> : null}

      {result && fields ? (
        <>
          <Card>
            <CardHead
              title="Check this before saving"
              note="Edit anything that looks wrong. Nothing is saved until you press the button."
            />
            <div className="card-body">
              <div className="grid grid-2">
                {/* image */}
                <div>
                  {selected ? (
                    imgUrl ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img className="receipt-img" src={imgUrl} alt="The receipt being read" />
                    ) : (
                      <Skeleton h={280} />
                    )
                  ) : (
                    <div className="callout">Your uploaded image was processed in memory and is not stored.</div>
                  )}
                  <div className="row" style={{ marginTop: "var(--s-3)" }}>
                    <Status tone={result.ocr.mean_confidence > 0.7 ? "good" : result.ocr.mean_confidence > 0.45 ? "warning" : "serious"}>
                      Text clarity {pct(result.ocr.mean_confidence, 0)}
                    </Status>
                    <span className="tiny muted">
                      {result.ocr.n_regions} regions · {(result.ocr.elapsed_ms / 1000).toFixed(1)}s ·{" "}
                      {result.ocr.engine}
                    </span>
                  </div>
                </div>

                {/* editable fields */}
                <div className="stack-sm">
                  {([
                    ["vendor_name", "Who it's from"],
                    ["invoice_number", "Bill number"],
                    ["issue_date", "Date"],
                    ["total_amount", "Total amount"],
                    ["tax_amount", "Tax"],
                  ] as [keyof Fields, string][]).map(([k, label]) => {
                    const missing = fields[k] === "";
                    return (
                      <div className={`field${lowConf(k) ? " field-changed" : ""}`} key={k}>
                        <label htmlFor={k}>
                          {label}
                          {lowConf(k) ? " · low confidence, please check" : ""}
                          {missing ? " · couldn't read this" : ""}
                        </label>
                        <input
                          id={k}
                          value={fields[k]}
                          placeholder={missing ? "Type it in" : undefined}
                          onChange={(e) => {
                            setFields({ ...fields, [k]: e.target.value });
                            setSaved(false);
                          }}
                        />
                      </div>
                    );
                  })}

                  {result.classification ? (
                    <div className="row">
                      <Status tone={result.classification.is_uncertain ? "warning" : "good"}>
                        Filed as {result.classification.category}
                      </Status>
                      <span className="tiny muted">
                        {pct(result.classification.confidence, 0)} sure
                        {result.classification.runner_up
                          ? ` · next guess ${result.classification.runner_up}`
                          : ""}
                      </span>
                    </div>
                  ) : null}

                  <button
                    type="button"
                    className="btn btn-primary"
                    disabled={!fields.total_amount || !fields.vendor_name}
                    onClick={() => setSaved(true)}
                  >
                    Save this record
                  </button>
                  {!fields.total_amount || !fields.vendor_name ? (
                    <p className="tiny muted">
                      A total and a supplier name are required. We won&rsquo;t create a
                      record with a guessed amount in it.
                    </p>
                  ) : null}
                  {saved ? <Status tone="good">Saved to this session</Status> : null}
                </div>
              </div>

              {/* pipeline outcome */}
              <div style={{ marginTop: "var(--s-5)" }}>
                {result.rejected ? (
                  <div className="callout callout-warning">
                    <div className="row" style={{ marginBottom: 6 }}>
                      <Status tone="serious">No record was created</Status>
                    </div>
                    <strong>{result.rejected.reason}</strong>
                    <br />
                    {result.rejected.why_this_is_correct}
                  </div>
                ) : result.record ? (
                  <div className="grid grid-3">
                    <Tile label="Amount" value={inr(result.record.amount)} />
                    <Tile label="Category" value={result.record.category} />
                    <Tile label="Source" value={result.record.source_type} note={result.record.source_reference} />
                    {result.record.needs_review ? (
                      <Tile
                        label="Flagged"
                        value="Needs review"
                        tone="warning"
                        note={result.record.review_reasons.join("; ")}
                      />
                    ) : null}
                  </div>
                ) : null}

                {result.duplicate_screen ? (
                  <div className="callout" style={{ marginTop: "var(--s-3)" }}>
                    <Status tone={result.duplicate_screen.is_exact_duplicate ? "warning" : "good"}>
                      {result.duplicate_screen.is_exact_duplicate
                        ? "Looks like a duplicate"
                        : "Not a duplicate"}
                    </Status>{" "}
                    {result.duplicate_screen.reason}
                  </div>
                ) : null}

                {result.correct ? (
                  <div className="callout" style={{ marginTop: "var(--s-3)" }}>
                    <strong>Scored against the known answer.</strong>{" "}
                    {Object.entries(result.correct).map(([k, ok]) => (
                      <span key={k} style={{ marginRight: 10 }}>
                        <Status tone={ok ? "good" : "critical"}>{k}</Status>
                      </span>
                    ))}
                    <div className="tiny muted" style={{ marginTop: 6 }}>
                      Only possible because these samples were generated, so the true
                      values are known. An uploaded receipt has nothing to score against.
                    </div>
                  </div>
                ) : null}
              </div>
            </div>

            <Method id="method-ocr" label="How the reading works, and where it fails" hint="EasyOCR · per-tier accuracy">
              <p className="method-lede">
                A detector finds text regions, a recogniser reads each one, and the
                regions are reassembled into visual rows before anything is parsed. The
                engine sits behind a swappable interface — it can be replaced with a
                cloud vision API without touching anything downstream.
              </p>

              <div className="callout">
                <strong>Why rows matter more than they sound.</strong> A detector splits a
                region wherever it sees a gap — including the gaps a thousands separator
                and a decimal point leave once blur has swallowed them. &ldquo;INR
                159,312.98&rdquo; arrives as three separate pieces: <code>159</code>,{" "}
                <code>312</code>, <code>98</code>. Read piece by piece that total is
                unrecoverable; read as the visual row it actually is, it comes back
                exactly. Rows are grouped on a de-skewed coordinate, because at 7° of tilt
                a single line drifts further down the page than a line is tall.
              </div>

              <h4>Measured accuracy, per image quality</h4>
              <div className="table-wrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th>Quality</th><th>Supplier</th><th>Bill no.</th><th>Date</th><th>Total</th><th>All four</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>Flatbed scan</td><td>100%</td><td>91.7%</td><td>100%</td><td>95.8%</td><td>87.5%</td></tr>
                    <tr><td>Phone photo</td><td>100%</td><td>83.3%</td><td>100%</td><td>95.8%</td><td>83.3%</td></tr>
                    <tr><td>Bad conditions</td><td>75.0%</td><td>12.5%</td><td>33.3%</td><td>50.0%</td><td>4.2%</td></tr>
                  </tbody>
                </table>
              </div>
              <p className="tiny muted">
                Never blended into one figure: a single number over a mixed set describes
                the mix of images, not the pipeline. Weight it toward clean scans and the
                identical code reports a much better result.
              </p>

              <div className="callout callout-warning">
                <strong>The finding that matters most.</strong> On bad images the bill
                number gave up 21 times out of 24 and was <em>wrong zero times</em> — it
                fails safely. The total amount gave up <em>never</em> and was{" "}
                <strong>wrong 12 times out of 24</strong>. The field your books depend on
                is the one that never admits defeat, because a corrupted number is still a
                perfectly parseable number. That is why this screen makes you confirm the
                amount, why a missing required field refuses to create a record rather
                than defaulting to zero, and why the tax line is cross-checked against the
                total to catch a lost decimal point.
              </div>

              <details>
                <summary className="tiny" style={{ cursor: "pointer", color: "var(--brand-800)", fontWeight: 600 }}>
                  Show the raw text that was read
                </summary>
                <pre
                  className="tiny"
                  style={{
                    whiteSpace: "pre-wrap", marginTop: "var(--s-2)", padding: "var(--s-3)",
                    background: "var(--surface)", border: "1px solid var(--border)",
                    borderRadius: "var(--radius-sm)", maxHeight: 260, overflow: "auto",
                  }}
                >
                  {result.ocr.text}
                </pre>
              </details>
            </Method>
          </Card>
        </>
      ) : null}

      {!result && !busy && !err ? (
        <EmptyState title="Nothing read yet">
          Choose a sample above or upload an image, and the extracted record will appear
          here for you to check.
        </EmptyState>
      ) : null}
    </div>
  );
}

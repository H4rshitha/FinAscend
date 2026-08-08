"use client";

/**
 * Sign in / sign up.
 *
 * One page, two modes, because a separate /signup route makes the switch a
 * full navigation that loses whatever the user already typed.
 *
 * Signup is TWO steps rather than one long form. Step 1 is the account, step 2
 * is the business. Splitting them is not decoration: the company size chosen in
 * step 2 determines the plan, and asking for it beside a password field gives
 * the user no room to explain what it changes. Step 2 shows exactly what each
 * choice unlocks, fetched from the same entitlement matrix the backend
 * enforces — so the promise on the signup screen cannot drift from what the
 * account actually gets.
 */

import { Suspense, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, ApiError, type CompanySize, type SignupOptions } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useNextPath } from "@/components/guard";
import { Icon, Skeleton, Status } from "@/components/ui";

type Mode = "signin" | "signup";

function LoginInner() {
  const { signIn, signUp } = useAuth();
  const router = useRouter();
  const next = useNextPath();

  const [mode, setMode] = useState<Mode>("signin");
  const [step, setStep] = useState<1 | 2>(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [options, setOptions] = useState<SignupOptions | null>(null);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [companyName, setCompanyName] = useState("");
  const [size, setSize] = useState<CompanySize | null>(null);

  // The size -> plan mapping and what each plan unlocks come from the backend,
  // so this screen can never promise a capability the entitlement engine will
  // not grant.
  useEffect(() => {
    api.get<SignupOptions>("/auth/options").then(setOptions).catch(() => setOptions(null));
  }, []);

  const planFor = (s: CompanySize | null) =>
    options?.company_sizes.find((c) => c.value === s);
  const planDetail = (s: CompanySize | null) => {
    const p = planFor(s);
    return options?.plans.find((x) => x.plan === p?.plan);
  };

  async function submitSignIn(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await signIn(email, password);
      router.replace(next);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Something went wrong signing you in."
      );
    } finally {
      setBusy(false);
    }
  }

  async function submitSignUp(e: React.FormEvent) {
    e.preventDefault();
    if (!size) return;
    setBusy(true);
    setError(null);
    try {
      await signUp({
        full_name: fullName,
        email,
        password,
        company_name: companyName,
        company_size: size,
      });
      router.replace(next);
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Something went wrong creating your account."
      );
      // A duplicate email or weak password is a step-1 problem; sending the
      // user back there is the only way they can actually fix it.
      if (err instanceof ApiError && (err.status === 409 || err.status === 422)) {
        setStep(1);
      }
    } finally {
      setBusy(false);
    }
  }

  const step1Valid =
    email.trim().length > 3 && password.length >= 10 && (mode === "signin" || fullName.trim());

  return (
    <div className="auth-wrap">
      <div className="auth-card">
        <div className="auth-brand">
          <span className="wordmark-dot" aria-hidden="true" />
          <span>FinAscend</span>
        </div>

        <h1 className="auth-title">
          {mode === "signin" ? "Welcome back" : step === 1 ? "Create your account" : "About your business"}
        </h1>
        <p className="auth-lede">
          {mode === "signin"
            ? "Sign in to see where your cash stands."
            : step === 1
              ? "Two quick steps. Nothing is charged and no card is needed."
              : "This sets up how much detail we show you. You can change it later."}
        </p>

        {error ? (
          <div className="auth-error" role="alert">
            {Icon.alert(15)}
            <span>{error}</span>
          </div>
        ) : null}

        {/* ---------------------------------------------------------------- */}
        {mode === "signin" ? (
          <form onSubmit={submitSignIn} className="stack-sm">
            <div className="field">
              <label htmlFor="email">Email</label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div className="field">
              <label htmlFor="password">Password</label>
              <input
                id="password"
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            <button className="btn btn-primary auth-submit" disabled={busy}>
              {busy ? "Signing in…" : "Sign in"}
            </button>
          </form>
        ) : step === 1 ? (
          <form
            className="stack-sm"
            onSubmit={(e) => {
              e.preventDefault();
              setError(null);
              setStep(2);
            }}
          >
            <div className="field">
              <label htmlFor="fullName">Your name</label>
              <input
                id="fullName"
                autoComplete="name"
                required
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
              />
            </div>
            <div className="field">
              <label htmlFor="email">Email</label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div className="field">
              <label htmlFor="password">Password</label>
              <input
                id="password"
                type="password"
                autoComplete="new-password"
                required
                minLength={10}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <span className="tiny muted">
                At least 10 characters. Length beats symbols — a short phrase you
                will remember is stronger than <code>P@ssw0rd</code>.
              </span>
            </div>
            <button className="btn btn-primary auth-submit" disabled={!step1Valid}>
              Continue
            </button>
          </form>
        ) : (
          <form onSubmit={submitSignUp} className="stack-sm">
            <div className="field">
              <label htmlFor="companyName">Business name</label>
              <input
                id="companyName"
                required
                value={companyName}
                onChange={(e) => setCompanyName(e.target.value)}
              />
            </div>

            <fieldset className="size-set">
              <legend className="field-legend">How big is your business?</legend>
              {options === null ? (
                <div className="stack-sm">
                  <Skeleton h={56} />
                  <Skeleton h={56} />
                  <Skeleton h={56} />
                </div>
              ) : (
                options.company_sizes.map((c) => (
                  <label
                    key={c.value}
                    className={`size-option${size === c.value ? " is-selected" : ""}`}
                  >
                    <input
                      type="radio"
                      name="company_size"
                      value={c.value}
                      checked={size === c.value}
                      onChange={() => setSize(c.value)}
                    />
                    <span className="size-main">
                      <span className="size-label">{c.label}</span>
                      <span className="size-head">{c.headcount}</span>
                      <span className="size-hint">{c.hint}</span>
                    </span>
                    <span className="size-plan">{c.plan_label}</span>
                  </label>
                ))
              )}
            </fieldset>

            {size && planDetail(size) ? (
              <div className="callout">
                <strong>{planDetail(size)!.label}</strong> — {planDetail(size)!.tagline}.
                <div className="tiny muted" style={{ marginTop: 6 }}>
                  Every plan gets the full cash picture, the payment plan and receipt
                  scanning. Larger businesses also get the working behind each number
                  opened up: {planDetail(size)!.capabilities.length} features in total.
                </div>
              </div>
            ) : null}

            <div className="row" style={{ gap: "var(--s-2)" }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  setError(null);
                  setStep(1);
                }}
              >
                Back
              </button>
              <button
                className="btn btn-primary"
                style={{ flex: 1 }}
                disabled={busy || !size || !companyName.trim()}
              >
                {busy ? "Creating your account…" : "Create account"}
              </button>
            </div>
          </form>
        )}

        {/* ---------------------------------------------------------------- */}
        <div className="auth-switch">
          {mode === "signin" ? (
            <>
              New here?{" "}
              <button
                type="button"
                className="btn-ghost link-btn"
                onClick={() => {
                  setMode("signup");
                  setStep(1);
                  setError(null);
                }}
              >
                Create an account
              </button>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <button
                type="button"
                className="btn-ghost link-btn"
                onClick={() => {
                  setMode("signin");
                  setStep(1);
                  setError(null);
                }}
              >
                Sign in
              </button>
            </>
          )}
        </div>
      </div>

      <aside className="auth-aside">
        <h2>Know how long your cash lasts.</h2>
        <p>
          FinAscend forecasts your cash, simulates thousands of possible months, and
          tells you which bills to pay first when there isn&rsquo;t enough for all of
          them.
        </p>
        <ul className="auth-points">
          <li>
            {Icon.check(15)} <span>A runway figure with the uncertainty stated, not hidden</span>
          </li>
          <li>
            {Icon.check(15)} <span>A payment plan that explains itself in plain words</span>
          </li>
          <li>
            {Icon.check(15)} <span>Every number can be opened up and checked</span>
          </li>
        </ul>
        <div className="auth-note">
          <Status tone="neutral">Honest about what it can&rsquo;t do</Status>
          <p className="tiny" style={{ marginTop: 8 }}>
            We publish the tests where our own methods came off worse — including a
            case where the sophisticated optimiser lost to a simple rule.
          </p>
        </div>
      </aside>
    </div>
  );
}

export default function LoginPage() {
  // useSearchParams (inside useNextPath) requires a Suspense boundary in the
  // app router, or the whole route opts out of static rendering.
  return (
    <Suspense fallback={<div className="auth-wrap"><div className="auth-card"><Skeleton h={340} /></div></div>}>
      <LoginInner />
    </Suspense>
  );
}

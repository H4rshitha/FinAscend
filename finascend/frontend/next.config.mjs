/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    // Single place the API base is configured. Read at build time into the
    // client bundle; override with NEXT_PUBLIC_API_BASE when the backend is
    // not on the default port.
    NEXT_PUBLIC_API_BASE:
      process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000/api/v1",
  },
};

export default nextConfig;

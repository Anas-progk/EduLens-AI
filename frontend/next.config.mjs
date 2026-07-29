/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  experimental: {
    workerThreads: true,
    webpackBuildWorker: false,
    cpus: 1,
  },

  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL}/api/:path*`,
      },
    ];
  },

  images: {
    domains: ["localhost"],
  },
};

export default nextConfig;
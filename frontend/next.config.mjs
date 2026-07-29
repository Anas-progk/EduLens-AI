/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  experimental: {
    workerThreads: true,
    webpackBuildWorker: false,
    cpus: 1,
  },


  images: {
    domains: ["localhost"],
  },
};

export default nextConfig;
import type { MetadataRoute } from 'next';

export default function Icon(): MetadataRoute.Icon {
  return {
    url: 'data:image/svg+xml;utf8,' +
      encodeURIComponent(
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
          <rect width="64" height="64" rx="14" fill="#3b82f6"/>
          <path d="M18 22h28v6H18zm0 12h28v6H18zm0 12h20v6H18z" fill="#ffffff"/>
        </svg>`
      )
  };
}

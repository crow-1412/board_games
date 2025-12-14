import './globals.css';
import type { ReactNode } from 'react';

export const metadata = {
  title: 'Board Games Tutor',
  description: '把复杂规则变成可执行教学流程（MVP: 阿瓦隆）'
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <div className="container">{children}</div>
      </body>
    </html>
  );
}

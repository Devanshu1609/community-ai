"use client"

import type React from 'react';

import { AuthGuard } from '@/components/auth-guard';
import { ThemeProvider } from '@/components/theme-provider';
import { SidebarProvider } from '@/components/ui/sidebar';
import { ToastProvider } from '@/components/ui/toast';

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <AuthGuard>
      <SidebarProvider>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <ToastProvider>
            {children}
          </ToastProvider>
        </ThemeProvider>
      </SidebarProvider>
    </AuthGuard>
  )
}

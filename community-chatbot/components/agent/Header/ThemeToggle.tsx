import { Moon, Sun } from 'lucide-react';
import { useTheme } from 'next-themes';

import { Button } from '@/components/ui/button';

export function ThemeToggle() {
    const { theme, setTheme } = useTheme()

    return (
        <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === "light" ? "dark" : "light")}
        >
            <Sun className="w-5 h-5 rotate-0 dark:-rotate-90 scale-100 dark:scale-0 transition-all" />
            <Moon className="absolute w-5 h-5 rotate-90 dark:rotate-0 scale-0 dark:scale-100 transition-all" />
            <span className="sr-only">Toggle theme</span>
        </Button>
    )
}

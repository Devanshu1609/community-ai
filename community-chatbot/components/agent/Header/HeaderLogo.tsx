import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';

export function HeaderLogo() {
    return (
        <Avatar className="w-8 sm:w-10 h-8 sm:h-10">
            <AvatarImage src="/mifos.png" alt="Mifos Logo" />
            <AvatarFallback className="bg-blue-600 text-white">M</AvatarFallback>
        </Avatar>
    );
}
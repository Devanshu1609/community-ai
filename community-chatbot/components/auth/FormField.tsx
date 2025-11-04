"use client";

import { useState } from 'react';

import { Eye, EyeOff } from 'lucide-react';
import { FieldValues } from 'react-hook-form';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { FormFieldProps, InputType } from '@/types/auth/types';

export function FormField<T extends FieldValues>({
	id,
	label,
	placeholder,
	type,
	icon,
	register,
	error,
}: FormFieldProps<T>) {
	const [showPassword, setShowPassword] = useState(false);

	let inputType: InputType;
	if (type === "password") {
		inputType = showPassword ? "text" : "password";
	} else {
		inputType = type;
	}

	return (
		<div className="space-y-2">
			<Label htmlFor={id}>
				{label}
			</Label>
			<div className="relative">
				{icon && <span className="top-3 left-3 absolute w-4 h-4 text-muted-foreground">{icon}</span>}

				<Input
					id={id}
					type={inputType}
					placeholder={placeholder}
					{...(register ? register(id) : {})}
					className={`pl-10 ${type === "password" ? "pr-10" : ""}`}
				/>

				{type === "password" && (
					<Button
						type="button"
						variant="ghost"
						size="sm"
						className="top-0 right-0 absolute hover:bg-transparent px-3 py-2 h-full"
						onClick={() => setShowPassword(!showPassword)}
					>
						{showPassword ? (
							<EyeOff className="w-4 h-4 text-muted-foreground" />
						) : (
							<Eye className="w-4 h-4 text-muted-foreground" />
						)}
					</Button>
				)}
			</div>

			{error && <p className="text-red-500 text-sm">{error}</p>}
		</div>
	);
}
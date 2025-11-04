import { FieldValues, Path, UseFormRegister } from 'react-hook-form';

export interface LayoutProps {
	title: string
	description: string
	children: React.ReactNode
}

export type InputType = "text" | "email" | "password";

export interface FormFieldProps<T extends FieldValues> {
	id: Path<T>;
	label: string;
	type: InputType;
	placeholder?: string;
	icon?: React.ReactNode;
	register?: UseFormRegister<T>;
	error?: string;
};
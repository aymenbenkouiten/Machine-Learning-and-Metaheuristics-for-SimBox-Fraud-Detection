<?php

namespace App\Models;

// use Illuminate\Contracts\Auth\MustVerifyEmail;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Foundation\Auth\User as Authenticatable;
use Illuminate\Notifications\Notifiable;

class User extends Authenticatable
{
    use HasFactory, Notifiable;

    /**
     * The attributes that are mass assignable.
     *
     * @var array<int, string>
     */
    protected $fillable = [
        'name',
        'email',
        'password',
        'role',
        'permissions',
        'created_by',
        'deleted_at'
    ];

    /**
     * The attributes that should be hidden for serialization.
     *
     * @var array<int, string>
     */
    protected $hidden = [
        'password',
        'remember_token',
    ];

    /**
     * Get the attributes that should be cast.
     *
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'email_verified_at' => 'datetime',
            'password' => 'hashed',
        ];
    }
    public function Created_by()
    {
        if($this->created_by != null)
        {
            return User::find($this->created_by);
        }
        return new User([
            'role' => '',
            'name' => 'n/a'
        ]);
    }
    public function Has_Permission($permissions)
    {
        if($this->permissions != null)
        {
            if(is_array($permissions))
            {
                foreach($permissions as $permission)
                {
                    if($this->Has_Permission($permission))
                    {
                        return true;
                    }
                }
            }
            else
            {
                return in_array($permissions ,explode(',', $this->permissions));
            }
        }
        return false;
    }
}

$args
$tag = $args[0]
if (!$tag -match "^(?<version>[0-9]+\.[0-9]+\.[0-9]+)(?<suffix>.*)?")
{
    "Invalid"
    throw "Invalide1"
}

$Matches
if (!$Matches.ContainsKey('version')) {
     throw "Invalid version in $tag"
}


if ($Matches.ContainsKey('suffix'))
{
    $Matches
    $suffix = $Matches['suffix']
    # -type is required. dotversion is not
    if ((!$suffix -match "\-(?<type>\w+)(?<dotversion>\.[0-9]+)?")  -or
        (!('alpha', 'beta', 'rc') -contains $Matches['type']) -or
        ($Matches.Length -eq 3 -and !$Matches.ContainsKey('dotversion')))
    {
        throw ("Invalid suffix $suffix")
    }
}

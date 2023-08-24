# Repilot Patch

```
 // this.advance();
```

# Developer Patch

```

```

# Context

```
--- bug/Jsoup-24/src/main/java/org/jsoup/parser/TokeniserState.java

+++ fix/Jsoup-24/src/main/java/org/jsoup/parser/TokeniserState.java

@@ -555,7 +555,7 @@

                 String name = r.consumeLetterSequence();
                 t.tagPending.appendTagName(name.toLowerCase());
                 t.dataBuffer.append(name);
-                r.advance();
+ // this.advance();
                 return;
             }
```

# Note

